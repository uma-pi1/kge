import torch
import time

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob, TrainingJob
from kge.model import KgeModel
from kge.job.trace import format_trace_entry

from typing import Dict, Optional, Any


class EvaluationJob(TrainingOrEvaluationJob):
    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("eval.batch_size")
        self.device = self.config.get("job.device")
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.eval_split = self.config.get("eval.split")
        self.epoch = -1

        # all done, run job_created_hooks if necessary
        if self.__class__ == EvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(config, dataset, parent_job=None, model=None):
        """Factory method to create an evaluation job """
        from kge.job import EntityRankingJob, EntityPairRankingJob

        # create the job
        if config.get("eval.type") == "entity_ranking":
            return EntityRankingJob(config, dataset, parent_job=parent_job, model=model)
        elif config.get("eval.type") == "entity_pair_ranking":
            return EntityPairRankingJob(
                config, dataset, parent_job=parent_job, model=model
            )
        elif config.get("eval.type") == "training_loss":
            return TrainingLossEvaluationJob(
                config, dataset, parent_job=parent_job, model=model
            )
        else:
            raise ValueError("eval.type")

    def _prepare(self):
        """Prepare this job for running.
        Guaranteed to be called exactly once before running the first epoch.

        """
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

    def _run(self) -> Dict[str, Any]:
        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )

        self._evaluate()

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in self.current_trace["epoch"]:
            self.current_trace["epoch"][metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **self.current_trace["epoch"]),
            )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("eval_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None

        # reset model and return metrics
        if was_training:
            self.model.train()

        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        return trace_entry

    def _evaluate(self):
        """
        Compute evaluation metrics, output results to trace file.
        The results of the evaluation must be written into self.current_trace["epoch"]
        """
        raise NotImplementedError

    def _load(self, checkpoint: Dict):
        if checkpoint["type"] not in ["train", "package"]:
            raise ValueError("Can only evaluate train and package checkpoints.")
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.epoch = checkpoint["epoch"]
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint["file"]
        )

    @classmethod
    def create_from(
        cls,
        checkpoint: Dict,
        new_config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
        eval_split: Optional[str] = None,
    ) -> Job:
        """
        Creates a Job based on a checkpoint
        Args:
            checkpoint: loaded checkpoint
            new_config: optional config object - overwrites options of config
                              stored in checkpoint
            dataset: dataset object
            parent_job: parent job (e.g. search job)
            eval_split: 'valid' or 'test'.
                        Defines the split to evaluate on.
                        Overwrites split defined in new_config or config of
                        checkpoint.

        Returns: Evaluation-Job based on checkpoint

        """
        if new_config is None:
            new_config = Config(load_default=False)
        if not new_config.exists("job.type") or new_config.get("job.type") != "eval":
            new_config.set("job.type", "eval", create=True)
        if eval_split is not None:
            new_config.set("eval.split", eval_split, create=True)

        return super().create_from(checkpoint, new_config, dataset, parent_job)


class TrainingLossEvaluationJob(EvaluationJob):
    """ Evaluating by using the training loss """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        training_loss_eval_config = config.clone()
        # TODO set train split to include validation data here
        #   once support is added
        #   Then reflect this change in the trace entries

        self._train_job = TrainingJob.create(
            config=training_loss_eval_config,
            parent_job=self,
            dataset=dataset,
            model=model,
            forward_only=True,
        )

        if self.__class__ == TrainingLossEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # prepare training job
        self._train_job._prepare()
        self._train_job._is_prepared = True

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, Any]:
        if self.parent_job:
            self.epoch = self.parent_job.epoch

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="training_loss_evaluation",
            scope="epoch",
            split=self._train_job.config.get("train.split"),
            epoch=self.epoch,
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # let's go
        train_trace_entry = self._train_job.run_epoch()

        # compute trace
        self.current_trace["epoch"].update(
            dict(
                epoch_time=train_trace_entry.get("epoch_time"),
                event="eval_completed",
                size=train_trace_entry["size"],
                avg_loss=train_trace_entry["avg_loss"],
                avg_penalty=train_trace_entry["avg_penalty"],
                avg_cost=train_trace_entry["avg_cost"],
            )
        )
