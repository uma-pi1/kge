from kge.model import ProjectionEmbedder
from kge.model.rescal import rescal_set_relation_embedder_dim


class Tucker3LookupEmbedder(ProjectionEmbedder):
    """A ProjectionEmbedder that expands relation embeddings to size entity_dim^2"""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        # TODO initialization
        # TODO dropout is not applied to core tensor, but only to mixing matrices
        rescal_set_relation_embedder_dim(config, dataset, configuration_key)
        super().__init__(config, dataset, configuration_key, vocab_size)

        # init_core_tensor_std = math.sqrt(variance) / (
        #     self.get_option("entity_dim")
        #     * math.sqrt(self.get_option("relation_dim"))
        #     * init_std ** 3
        # )
        # torch.nn.init.normal_(
        #     self.relation_projection.weight.data, init_core_tensor_std
        # )
