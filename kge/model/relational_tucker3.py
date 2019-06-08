import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel, KgeEmbedder, KgeBase
from kge.model.rescal import RESCALScorer


class RelationalTucker3(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(
            config,
            dataset,
            scorer=RESCALScorer(config=config, dataset=dataset))

        config.set(
            config.get("model") + ".relation_embedder.entity_dim",
            config.get(config.get("model") + ".entity_embedder.dim"),
            create=True,
        )

        config.set(
            config.get("model") + ".relation_embedder.entity_initialize_arg",
            config.get(config.get("model") + ".entity_embedder.initialize_arg"),
            create=True,
        )

        #: Embedder used for relations
        self._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            config.get("model") + ".relation_embedder",
            dataset.num_relations,
        )
