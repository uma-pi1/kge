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
            self.encoder = torch.nn.Conv1d(in_channels=99, out_channels=99, kernel_size=2, dilation=1, bias=False)
        if "entity" in self.configuration_key:  # Todo: in_channels = batch_size - (n_gram_size - 1)
            self.encoder = torch.nn.Conv1d(in_channels=99, out_channels=99, kernel_size=2, dilation=1, bias=False)

    #dataset.max_tokens_per_entity()

    def _pooling(self, token_embeddings, token_indexes):
        # pooling on token embeddings
        if self.pooling == 'max':
            pass
        elif self.pooling == 'mean':
            pass
        elif self.pooling == 'sum':
            pass
        else:
            raise NotImplementedError
        return None

    def embed(self, indexes: Tensor) -> Tensor:
        import time
        n_gram_size = 2

        '''
        import torch
        t1 = time.time()
        test = [indexes[i:i + n_gram_size] for i in range(len(indexes) - n_gram_size + 1)]
        t2 = time.time()
        print("T:", t2-t1)
        '''

        from nltk import ngrams
        t3 = time.time()
        test2 = list(ngrams(indexes, n_gram_size))
        t4 = time.time()
        print("T2:", t4-t3)

        gram_token_indexes = self.embed(test2)#.float()

        token_mask = (gram_token_indexes > 0).float()

        # Todo: Embed whole batch: batch_size x n_gram_size x max_tokens
        # embedding individual entries works:
        #self._embeddings(gram_token_indexes[0].long())

        # But not embedding the dimensionality reduced tensor
        # https://stackoverflow.com/questions/47205762/embedding-3d-data-in-pytorch
        #gram_token_indexes.long().view(-1, gram_token_indexes.size(2))


        size = gram_token_indexes.size()
        encoded_tokens = self.encoder(gram_token_indexes.reshape(size[2], size[0], size[1]))
        encoded_tokens = encoded_tokens.squeeze().transpose(0,1)
        #torch.transpose(encoded_tokens, 0, 1).squeeze()
        #.transpose(0, 2)).transpose(1,2)

        # Todo: mask 0 tokens
        x = 0

        token_indexes = self._token_lookup[indexes]
        pass
        #return self._pooling(super().embed(indexes), self._token_lookup[indexes])

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        x = 0
        pass
        #return self._pooling(super().embed_all(), self._token_lookup)