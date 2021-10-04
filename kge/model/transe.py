import torch
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp_":
            # we do not use matrix multiplication due to this issue
            # https://github.com/pytorch/pytorch/issues/42479
            out = -torch.cdist(
                s_emb + p_emb,
                o_emb,
                p=self._norm,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        elif combine == "_po":
            out = -torch.cdist(
                o_emb - p_emb,
                s_emb,
                p=self._norm,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling

        if (
            isinstance(job, TrainingJobNegativeSampling)
            and job.config.get("negative_sampling.implementation") == "auto"
        ):
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
