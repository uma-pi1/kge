import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel

from typing import Dict, Union, Optional


class EvaluationJob(Job):
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("eval.batch_size")
        self.device = self.config.get("job.device")
        max_k = min(
            self.dataset.num_entities(), max(self.config.get("entity_ranking.hits_at_k_s"))
        )
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("entity_ranking.hits_at_k_s"))
        )
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.eval_split = self.config.get("eval.split")
        self.filter_splits = self.config.get("entity_ranking.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)
        self.filter_with_test = config.get("entity_ranking.filter_with_test")
        self.epoch = -1

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks = []

        #: Signature: job, trace_entry
        self.post_valid_hooks = []

        #: Whether to create additional histograms for head and tail slot
        self.head_and_tail = config.get("entity_ranking.metrics_per.head_and_tail")

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: job, trace_entry
        self.hist_hooks = [hist_all]
        if config.get("entity_ranking.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("entity_ranking.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

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
        else:
            raise ValueError("eval.type")

    def _run(self) -> dict:
        """ Compute evaluation metrics, output results to trace file """
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


# HISTOGRAM COMPUTATION ###############################################################


def __initialize_hist(hists, key, job):
    """If there is no histogram with given `key` in `hists`, add an empty one."""
    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities()],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    """Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
    `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
    rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

    """
    __initialize_hist(hists, "all", job)
    if job.head_and_tail:
        __initialize_hist(hists, "head", job)
        __initialize_hist(hists, "tail", job)
        hist_head = hists["head"]
        hist_tail = hists["tail"]

    hist = hists["all"]
    for r in o_ranks:
        hist[r] += 1
        if job.head_and_tail:
            hist_tail[r] += 1
    for r in s_ranks:
        hist[r] += 1
        if job.head_and_tail:
            hist_head[r] += 1


def hist_per_relation_type(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    for rel_type, rels in job.dataset.index("relations_per_type").items():
        __initialize_hist(hists, rel_type, job)
        hist = hists[rel_type]
        if job.head_and_tail:
            __initialize_hist(hists, f"{rel_type}_head", job)
            __initialize_hist(hists, f"{rel_type}_tail", job)
            hist_head = hists[f"{rel_type}_head"]
            hist_tail = hists[f"{rel_type}_tail"]

        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(o_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_tail[r] += 1

        for r, m in zip(s_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_head[r] += 1


def hist_per_frequency_percentile(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    # initialize
    frequency_percs = job.dataset.index("frequency_percentiles")
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)

    # go
    for perc in frequency_percs["subject"].keys():  # same for relation and object
        for r, m_s, m_r in zip(
            s_ranks,
            [id in frequency_percs["subject"][perc] for id in s.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format("subject", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
        for r, m_o, m_r in zip(
            o_ranks,
            [id in frequency_percs["object"][perc] for id in o.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_o:
                hists["{}_{}".format("object", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
