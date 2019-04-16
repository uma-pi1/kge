import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder


class LookupEmbedder(KgeEmbedder):
    def __init__(self, config, dataset, configuration_key, is_entity_embedder):
        super().__init__(config, dataset, configuration_key, is_entity_embedder)

        ## read config
        self.dropout = self.get_option('dropout')
        self.dim = self.get_option('dim')
        self.sparse = self.get_option('sparse')
        self.normalize = self.get_option('normalize')
        self.size = dataset.num_entities if self.is_entity_embedder \
            else dataset.num_relations

        ## setup embedder
        self.embeddings = torch.nn.Embedding(self.size, self.dim,
                                             sparse=self.sparse)
        self.initialize(self.embeddings.weight.data,
                        self.get_option('initialize'),
                        self.get_option('initialize_arg'))

        ## TODO L2

    def _embed(self, embeddings):
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training)
        if self.normalize == 'L2':
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def embed(self, indexes):
        return self._embed(self.embeddings(indexes.long()))

    def embed_all(self):
        return self._embed(self.embeddings.weight)
