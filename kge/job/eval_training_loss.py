import torch

from kge import Config, Dataset
from kge.job import Job, EvaluationJob, TrainingJob

from typing import Any, Dict


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
