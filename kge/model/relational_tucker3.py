import math
import torch.nn
from kge import Config, Dataset
from kge.model.kge_model import KgeModel
from kge.model.rescal import RescalScorer, rescal_set_relation_embedder_dim
from kge.model import ProjectionEmbedder
from kge.misc import round_to_points


class RelationalTucker3(KgeModel):
    r"""Implementation of the Relational Tucker3 KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
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

        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RescalScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)
