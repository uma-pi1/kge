from kge import Config, Dataset
from kge.model.kge_model import KgeModel
from kge.model.rescal import RescalScorer, rescal_set_relation_embedder_dim


class RelationalTucker3(KgeModel):
    r"""Implementation of the Relational Tucker3 KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        rescal_set_relation_embedder_dim(
            config, dataset, config.get("model") + ".relation_embedder"
        )
        super().__init__(
            config, dataset, scorer=RescalScorer(config=config, dataset=dataset)
        )
