import torch.nn
from kge.util.l0module import _L0Norm_orig
from kge.model import Tucker3RelationEmbedder
from torch.nn import functional as F


class SparseTucker3RelationEmbedder(Tucker3RelationEmbedder):
    """Like Tucker3RelationEmbedder but with L0 penalty on projection (core tensor)"""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key, vocab_size)

        self.l0_weight = self.get_option("l0_weight")
        self.l0norm = _L0Norm_orig(
            self.projection,
            loc_mean=self.get_option("loc_mean"),
            loc_sdev=self.get_option("loc_sdev"),
            beta=self.get_option("beta"),
            gamma=self.get_option("gamma"),
            zeta=self.get_option("zeta"),
        )

        self.mask = None  # recomputed at each batch during training, else kept constant
        self.l0_penalty = None
        self.density = None

    def _embed(self, embeddings):
        if self.mask is None:
            self.mask, self.l0_penalty = self.l0norm._get_mask()
            with torch.no_grad():
                self.density = (
                    (self.mask > 0).float().sum() / self.mask.numel()
                ).item()

        embeddings = F.linear(
            embeddings, self.projection.weight * self.mask, self.projection.bias
        )
        if self.dropout > 0:
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        if self.normalize == "L2":
            embeddings = F.normalize(embeddings)
        return embeddings

    def _invalidate_mask(self):
        self.mask = None
        self.density = None
        self.l0_penalty = None

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + [self.l0_weight * self.l0_penalty]

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)

        # during training, we recompute the mask in every batch
        from kge.job import TrainingJob

        if isinstance(job, TrainingJob):
            job.pre_batch_hooks.append(lambda job: self._invalidate_mask())

        # append density to traces
        def append_density(job, trace):
            trace["core_tensor_density"] = self.density

        if isinstance(job, TrainingJob):
            job.post_batch_trace_hooks.append(append_density)
        job.post_epoch_trace_hooks.append(append_density)
