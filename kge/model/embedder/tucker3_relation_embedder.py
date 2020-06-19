from kge.model import ProjectionEmbedder
from kge.model.rescal import rescal_set_relation_embedder_dim


class Tucker3RelationEmbedder(ProjectionEmbedder):
    """A ProjectionEmbedder that expands relation embeddings to size entity_dim^2"""

    def __init__(
        self, config, dataset, configuration_key, vocab_size, init_for_load_only=False
    ):
        # TODO dropout is not applied to core tensor, but only to mixing matrices
        rescal_set_relation_embedder_dim(config, dataset, configuration_key)
        super().__init__(
            config,
            dataset,
            configuration_key,
            vocab_size,
            init_for_load_only=init_for_load_only,
        )
