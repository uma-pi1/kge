from torch import Tensor
import torch.nn

from kge import Config, Dataset
from kge.model import MentionEmbedder


class LstmLookupEmbedder(MentionEmbedder):

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

        self._dimensions = self.get_option("dim")
        self._encoder_lstm = torch.nn.LSTM(input_size=self._dimensions, hidden_size=self._dimensions,
                                           batch_first=True, dropout=0)

    def _forward(self, token_embeddings, token_indexes):
        # switch batch and sequence dimension to match input format
        lstm_input = token_embeddings.permute(1, 0, 2)
        lstm_output, hn = self._encoder_lstm(lstm_input)
        num_tokens = (token_indexes > 0).sum(dim=1)
        # Use Output after last token instead of sequence end
        #test = lstm_output.permute(1, 0, 2)
        #test2 = test[torch.arange(test.size(0)), num_tokens - 1]
        #return test2
        return lstm_output[token_indexes[0].size()[0] - 1, :, :]

    def embed(self, indexes: Tensor) -> Tensor:
        return self._forward(super().embed(indexes), self._token_lookup[indexes])

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        return self._forward(super().embed_all(), self._token_lookup)