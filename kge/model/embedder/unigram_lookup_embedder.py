from torch import Tensor

from kge import Config, Dataset
from kge.model import MentionEmbedder


class UnigramLookupEmbedder(MentionEmbedder):

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key: str,
            vocab_size: int,
            init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)
        self.pooling = self.get_option("pooling")

    def _pooling(self, token_embeddings, token_indexes):
        # pooling on token embeddings
        if self.pooling == 'max':  # should reduce dimensions to (batch_size, dim)
            pooled_embeddings = token_embeddings.max(dim=1).values
        elif self.pooling == 'mean':
            lengths = (token_indexes > 0).sum(dim=1)
            pooled_embeddings = token_embeddings.sum(dim=1) / lengths.unsqueeze(1)
        elif self.pooling == 'sum':
            pooled_embeddings = token_embeddings.sum(dim=1)
        else:
            raise NotImplementedError
        return pooled_embeddings

    def embed(self, indexes: Tensor) -> Tensor:
        return self._pooling(super().embed(indexes), self._token_lookup)

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        return self._pooling(super().embed_all(), self._token_lookup)