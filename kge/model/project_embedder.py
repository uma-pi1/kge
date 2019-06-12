import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder


class ProjectEmbedder(KgeEmbedder):
    """Adds a linear projection layer to a base embedder."""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key)

        # initialize base_embedder
        self.base_embedder = KgeEmbedder.create(
            config, dataset, configuration_key + ".base_embedder", vocab_size
        )

        # initialize projection
        if self.dim < 0:
            self.dim = self.base_embedder.dim
        self.dropout = self.get_option("dropout")
        self.normalize = self.get_option("normalize")
        self.check_option("normalize", ["", "L2"])
        self.projection = torch.nn.Linear(
            self.base_embedder.dim,
            self.dim,
            bias=False,
        )
        self.initialize(
            self.projection.weight.data,
            self.get_option("initialize"),
            self.get_option("initialize_arg"),
        )

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )
        if self.normalize == "L2":
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def embed(self, indexes):
        return self._embed(self.base_embedder.embed(indexes))

    def embed_all(self):
        return self._embed(self.base_embedder.embed_all())
