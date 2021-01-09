from torch import Tensor
from kge import Config, Dataset
from kge.model import MentionEmbedder
import torch

class BigramLookupEmbedder(MentionEmbedder):

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

        if "relation" in self.configuration_key:
            self.encoder = torch.nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=2, dilation=1, bias=False)
        if "entity" in self.configuration_key:
            self.encoder = torch.nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=2, dilation=1, bias=False)

    def _pooling(self, token_embeddings):
        # pooling on token embeddings
        if self.pooling == 'max':
            pooled = token_embeddings.max(dim=2).values
        elif self.pooling == 'sum':
            pooled = token_embeddings.sum(dim=2)
        else:
            raise NotImplementedError
        return pooled

    def embed(self, indexes: Tensor) -> Tensor:
        token_indexes = self.lookup_tokens(indexes)
        token_mask = (token_indexes > 0).unsqueeze(1).float()[:, :, 1:]
        token_embeddings = self.embed_tokens(token_indexes).transpose(1,2)
        encoded = self.encoder(token_embeddings)
        encoded = encoded + token_embeddings[:, :, 1:]
        pooled = self._pooling(encoded * token_mask)
        return pooled

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        token_indexes = self._token_lookup
        token_mask = (token_indexes > 0).unsqueeze(1).float()[:, :, 1:]
        token_embeddings = self.embed_tokens(token_indexes).transpose(1, 2)
        encoded = self.encoder(token_embeddings)
        encoded = encoded + token_embeddings[:, :, 1:]
        pooled = self._pooling(encoded * token_mask)
        return pooled
