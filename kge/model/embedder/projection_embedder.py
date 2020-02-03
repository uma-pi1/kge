import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder


class ProjectionEmbedder(KgeEmbedder):
    """Adds a linear projection layer to a base embedder."""

    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key)

        # initialize base_embedder
        if self.configuration_key + ".base_embedder.type" not in config.options:
            config.set(
                self.configuration_key + ".base_embedder.type",
                self.get_option("base_embedder.type"),
            )
        self.base_embedder = KgeEmbedder.create(
            config, dataset, self.configuration_key + ".base_embedder", vocab_size
        )

        # initialize projection
        if self.dim < 0:
            self.dim = self.base_embedder.dim
        self.dropout = self.get_option("dropout")
        self.normalize = self.check_option("normalize", ["", "L2"])
        self.regularize = self.check_option("regularize", ["", "l1", "l2"])
        self.projection = torch.nn.Linear(self.base_embedder.dim, self.dim, bias=False)
        self.initialize(
            self.projection.weight.data,
            self.get_option("initialize"),
            self.get_option("initialize_args"),
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

    def penalty(self, **kwargs):
        # TODO factor out to a utility method
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            p = []
        elif self.regularize == "l1":
            p = [
                self.get_option("regularize_weight")
                * self.projection.weight.norm(p=1)
            ]
        elif self.regularize == "l2":
            p = [
                self.get_option("regularize_weight")
                * self.projection.weight.norm(p=2) ** 2
            ]
        else:
            raise ValueError("unknown penalty")

        return super().penalty(**kwargs) + p + self.base_embedder.penalty(**kwargs)
