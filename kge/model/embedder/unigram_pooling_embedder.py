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
        self.pooling = self.get_option("pooling")  # TODO: add several pooling options such as 'sum', max, etc.
        # TODO: Add entity and relation dropout later
        #self.entity_dropout = entity_dropout if entity_dropout else dropout
        #self.relation_dropout = relation_dropout if relation_dropout else dropout

        '''
            if self.pool == 'max':
            elif self.pool == 'mean':
            if self.normalize == 'norm':
            if self.normalize == 'batchnorm':
        '''


        # each token is mapped to an n-dimensional embedding vector
        num_tokens = None   # TODO: get number of tokens
        self._embeddings = torch.nn.Embedding(num_tokens, self.dim, sparse=self.sparse,)


        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)