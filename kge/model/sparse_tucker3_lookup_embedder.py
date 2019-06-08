import math

import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder
from kge.util.l0module import L0Linear_orig


class SparseTucker3LookupEmbedder(KgeEmbedder):
    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key)

        # read config
        self.dropout = self.get_option("dropout")
        self.core_tensor_dropout = self.get_option("core_tensor_dropout")
        self.sparse = self.get_option("sparse")
        self.normalize = self.get_option("normalize")
        self.vocab_size = vocab_size

        # setup embedder
        self.embeddings = torch.nn.Embedding(
            self.vocab_size, self.get_option("relation_dim"), sparse=self.sparse
        )
        self.initialize(
            self.embeddings.weight.data,
            self.get_option("initialize"),
            self.get_option("initialize_arg"),
        )

        init_std = self.get_option("entity_initialize_arg")
        variance = self.get_option("variance")

        self.relation_projection = L0Linear_orig(
            self.get_option("relation_dim"), self.get_option("entity_dim") ** 2,
            loc_mean=self.get_option("loc_mean"),
            loc_sdev=self.get_option("loc_sdev"), beta=self.get_option("beta"),
            gamma=self.get_option("gamma"), zeta=self.get_option("zeta"),
            bias=False)
        init_core_tensor_std = \
            math.sqrt(variance)/(self.get_option("entity_dim")*
            math.sqrt(self.get_option("relation_dim"))*init_std**3)
        torch.nn.init.normal_(self.relation_projection._origin.weight.data,
                              init_core_tensor_std)
        self.penalty = None
        self.density = None

    def _embed(self, embeddings):
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )
        embeddings, self.penalty, self.density = self.relation_projection(embeddings)
        if self.core_tensor_dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.core_tensor_dropout, training=self.training
            )
        if self.normalize == "L2":
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def embed(self, indexes):
        return self._embed(self.embeddings(indexes.long()))

    def embed_all(self):
        return self._embed(self.embeddings.weight)
