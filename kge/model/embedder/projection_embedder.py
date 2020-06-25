import torch.nn
import torch.nn.functional
from kge.model import KgeEmbedder


class ProjectionEmbedder(KgeEmbedder):
    """Adds a linear projection layer to a base embedder."""

    def __init__(
        self, config, dataset, configuration_key, vocab_size, init_for_load_only=False
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

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
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.projection = torch.nn.Linear(self.base_embedder.dim, self.dim, bias=False)
        if not init_for_load_only:
            self._init_embeddings(self.projection.weight.data)

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        if self.dropout > 0:
            embeddings = torch.nn.functional.dropout(
                embeddings, p=self.dropout, training=self.training
            )
        return embeddings

    def embed(self, indexes):
        return self._embed(self.base_embedder.embed(indexes))

    def embed_all(self):
        return self._embed(self.base_embedder.embed_all())

    def penalty(self, **kwargs):
        # TODO factor out to a utility method
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            result = []
        elif self.regularize == "lp":
            p = self.get_option("regularize_args.p")
            result = [
                (
                    f"{self.configuration_key}.L{p}_penalty",
                    self.get_option("regularize_weight")
                    * self.projection.weight.norm(p=p).sum(),
                )
            ]
        else:
            raise ValueError("unknown penalty")

        return super().penalty(**kwargs) + result + self.base_embedder.penalty(**kwargs)
