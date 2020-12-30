from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

class UnigramPoolingEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only)

        self.dim = self.get_option("dim")
        self.sparse = self.get_option("sparse")
        self.pooling = self.get_option("pooling")
        # TODO: Add entity and relation dropout later
        #self.entity_dropout = entity_dropout if entity_dropout else dropout
        #self.relation_dropout = relation_dropout if relation_dropout else dropout

        if "relation" in self.configuration_key: # Todo: remove + 2 when tokens BOF (1) and EOF (2) are removed from dataset
            self._embeddings = torch.nn.Embedding(self.dataset.num_tokens_relations(), self.dim + 2, sparse=self.sparse)
        elif "entity" in self.configuration_key:
            self._embeddings = torch.nn.Embedding(self.dataset.num_tokens_entities(), self.dim + 2, sparse=self.sparse)
        #self._embeddings = torch.nn.Embedding(vocab_size, self.dim, sparse=self.sparse)


        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

        # Training:
        # input: quintuples s,r,o, s_a,o_a (indices)
        # s,r,o -> s,r,o tokens
        # pooling (max, sum,...)
        # token embeddings

        # embed input: subject indices
        # How to distinguish entity and relation embedder?

    def map_to_tokens(self, indexes: Tensor) -> Tensor:
        if "relation" in self.configuration_key:
            #token_indexes = self.dataset._mentions_to_token_ids['relations'][indexes]
            token_indexes = self.dataset.relation_mentions_to_token_ids()[indexes]
        elif "entity" in self.configuration_key:
            token_indexes = self.dataset.entity_mentions_to_token_ids()[indexes]
            #token_indexes = self.dataset._mentions_to_token_ids['entities'][indexes]
        return token_indexes

    def embed(self, indexes: Tensor) -> Tensor:
        # Todo: check if 1 and 2 are omitted beforehand (in OLPDataset)
        token_indexes = self.map_to_tokens(indexes) #[:,1:-1] # How to handle beginning and end for a relation([1,...2])? Omit for now
        # lookup all tokens -> token embeddings with
        # expected shape: 3D tensor (batch_size, max_tokens, dim=100)
        token_embeddings = self._embeddings(token_indexes)

        # pooling on token embeddings
        if self.pooling == 'max':  # should reduce dimensions to (batch_size, dim)
            token_embeddings = token_embeddings.max(dim=1).values
        elif self.pooling == 'mean':
            lengths = (token_indexes > 0).sum(dim=1)
            token_embeddings = token_embeddings.sum(dim=1) / lengths.unsqueeze(1)
        elif self.pooling == 'sum':
            token_embeddings.sum(dim=1)
        else:
            raise NotImplementedError

        #if self.normalize == 'norm':
        #    token_embeddings = torch.nn.functional.normalize(token_embeddings, dim=1) # l_p normalization
        #if self.normalize == 'batchnorm':
        #if dropout > 0:
        return token_embeddings

    def embed_all(self) -> Tensor:
        return None