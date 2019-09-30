import math
import torch.nn
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
from kge.model.rescal import RescalScorer, rescal_set_relation_embedder_dim
from kge.model import ProjectionEmbedder
from kge.util.misc import round_to_points


class Tucker3RelationEmbedder(ProjectionEmbedder):
    """A ProjectionEmbedder that expands relation embeddings to size entity_dim^2"""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        # TODO dropout is not applied to core tensor, but only to mixing matrices
        rescal_set_relation_embedder_dim(config, dataset, configuration_key)
        super().__init__(config, dataset, configuration_key, vocab_size)


class RelationalTucker3(KgeModel):
    r"""Implementation of the Relational Tucker3 KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        ent_emb_dim = self.get_option("entity_embedder.dim")
        ent_emb_conf_key = self.configuration_key + ".entity_embedder"
        round_ent_emb_dim_to = self.get_option("entity_embedder.round_dim_to")
        if len(round_ent_emb_dim_to) > 0:
            ent_emb_dim = round_to_points(round_ent_emb_dim_to, ent_emb_dim)
        config.set(ent_emb_conf_key + ".dim", ent_emb_dim, log=True)

        rescal_set_relation_embedder_dim(
            config, dataset, self.configuration_key + ".relation_embedder"
        )
        if (
            self.get_option("entity_embedder.initialize") == "auto_initialization"
            and self.get_option("relation_embedder.base_embedder.initialize")
            == "auto_initialization"
        ):
            dim_e = self.get_option("entity_embedder.dim")
            dim_r = self.get_option("relation_embedder.base_embedder.dim")
            dim_core = self.get_option("relation_embedder.dim")

            # entity embeddings -> unit norm in expectation
            config.set(
                self.configuration_key + ".entity_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".entity_embedder.initialize_args",
                {"mean": 0.0, "std": 1.0 / math.sqrt(dim_e)},
                log=True,
            )

            # relation embeddings -> unit norm in expectation
            config.set(
                self.configuration_key + ".relation_embedder.base_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key
                + ".relation_embedder.base_embedder.initialize_args",
                {"mean": 0.0, "std": 1.0 / math.sqrt(dim_r)},
                log=True,
            )

            # core tensor weight -> initial scores have var=1 (when no dropout / eval)
            config.set(
                self.configuration_key + ".relation_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".relation_embedder.initialize_args",
                {"mean": 0.0, "std": 1.0},
                log=True,
            )
        elif (
            self.get_option("entity_embedder.initialize") == "auto_initialization"
            or self.get_option("relation_embedder.base_embedder.initialize")
            == "auto_initialization"
        ):
            raise ValueError(
                "Both entity and relation embedders must be set to auto_initialization "
                "in order to use it."
            )

        super().__init__(
            config,
            dataset,
            scorer=RescalScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
        )

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)

        # append number of active parameters to trace
        from kge.model import SparseTucker3RelationEmbedder

        if isinstance(self.get_p_embedder(), SparseTucker3RelationEmbedder):

            def update_num_parameters(job, trace):
                # set by hook from superclass
                npars = trace["num_parameters"]

                # do not count mask as a parameter
                mask = self.get_p_embedder().mask
                npars -= mask.numel()
                trace["num_parameters"] = npars

                # do not count zeroed out connections
                with torch.no_grad():
                    npars -= (mask == 0).float().sum().item()
                    trace["num_active_parameters"] = int(npars)

            job.post_epoch_trace_hooks.append(update_num_parameters)
