import math
import torch.nn
from torch.nn import functional as F
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
from kge.model.rescal import RescalScorer, rescal_set_relation_embedder_dim
from kge.model import ProjectionEmbedder
from kge.model.rescal import rescal_set_relation_embedder_dim
from kge.util.l0module import _L0Norm_orig


class Tucker3RelationEmbedder(ProjectionEmbedder):
    """A ProjectionEmbedder that expands relation embeddings to size entity_dim^2"""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        # TODO dropout is not applied to core tensor, but only to mixing matrices
        rescal_set_relation_embedder_dim(config, dataset, configuration_key)
        super().__init__(config, dataset, configuration_key, vocab_size)


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
            self.density = ((self.mask > 0).float().sum() / self.mask.numel()).item()

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

        def append_density(job, trace):
            trace["core_tensor_density"] = self.density

        from kge.job import TrainingJob

        if isinstance(job, TrainingJob):
            # during training, we recompute the mask in every batch
            job.pre_batch_hooks.append(lambda job: self._invalidate_mask())
            job.post_batch_trace_hooks.append(append_density)

        job.post_epoch_trace_hooks.append(append_density)


class RelationalTucker3(KgeModel):
    r"""Implementation of the Relational Tucker3 KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        # The following is same behaviour as get_option from KgeModel
        # but no object yet at this point
        if configuration_key:
            rescal_set_relation_embedder_dim(
                config,
                dataset,
                config.get(configuration_key + ".model") + ".relation_embedder"
            )
        else:
            rescal_set_relation_embedder_dim(
                config,
                dataset,
                config.get("model") + ".relation_embedder"

        if config.get("relational_tucker3.auto_initialization"):
            dim_e = config.get_first(
                config.get("model") + ".entity_embedder.dim",
                config.get(config.get("model") + ".entity_embedder.type") + ".dim",
            )
            dim_r = config.get_first(
                config.get("model") + ".relation_embedder.base_embedder.dim",
                config.get_first(
                    config.get("model") + ".relation_embedder.base_embedder.type",
                    config.get(config.get("model") + ".relation_embedder.type")
                    + ".base_embedder.type",
                )
                + ".dim",
            )
            dim_core = config.get(config.get("model") + ".relation_embedder.dim")

            # entity embeddings -> unit norm in expectation
            config.set(
                config.get("model") + ".entity_embedder.initialize", "normal", log=True
            )
            config.set(
                config.get("model") + ".entity_embedder.initialize_arg",
                1.0 / math.sqrt(dim_e),
                log=True,
            )

            # relation embeddings -> unit norm in expectation
            config.set(
                config.get("model") + ".relation_embedder.base_embedder.initialize",
                "normal",
                log=True,
            )
            config.set(
                config.get("model") + ".relation_embedder.base_embedder.initialize_arg",
                1.0 / math.sqrt(dim_r),
                log=True,
            )

            # core tensor weight -> initial scores have var=1 (when no dropout / eval)
            config.set(
                config.get("model") + ".relation_embedder.initialize",
                "normal",
                log=True
            )
            config.set(
                config.get("model") + ".relation_embedder.initialize_arg",
                1.0,
                log=True
>>>>>>> daecbfac63f163dc7e146e8fbf72ea6cbf6ecd75
            )

        super().__init__(
            config,
            dataset,
            scorer=RescalScorer(config=config, dataset=dataset),
            configuration_key=configuration_key
        )
