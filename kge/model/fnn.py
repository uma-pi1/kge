import torch
import torch.nn as nn
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util.misc import get_activation_function
from collections import OrderedDict


class FnnScorer(RelationalScorer):
    r"""Implementation of the ComplEx KGE scorer."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        subject_dim: int,
        relation_dim: int,
        object_dim: int,
    ):
        super().__init__(config, dataset)

        hidden_layers_size = self.config.get("fnn.hidden_layers.size")
        if hidden_layers_size < 0:
            hidden_layers_size = subject_dim
        dropout = self.config.get("fnn.hidden_layers.dropout")
        layers = OrderedDict()
        last_size = subject_dim + relation_dim + object_dim
        for i in range(self.config.get("fnn.hidden_layers.number")):  # hidden layers
            layer = nn.Linear(last_size, hidden_layers_size)
            last_size = hidden_layers_size
            self.initialize(
                layer.weight,
                self.config.get("fnn.hidden_layers.initialize"),
                self.config.get("fnn.hidden_layers.initialize_arg"),
            )
            layers["linear" + str(i + 1)] = layer
            layers["nonlinear" + str(i + 1)] = get_activation_function(
                self.config.get("fnn.hidden_layers.activation")
            )
            if dropout > 0.0:  # note: input dropout handled separately by embedder
                layers["dropout" + str(i + 1)] = nn.Dropout(dropout)

        layers["output"] = nn.Linear(last_size, 1)
        self.initialize(
            layers["output"].weight,
            self.config.get("fnn.hidden_layers.initialize"),
            self.config.get("fnn.hidden_layers.initialize_arg"),
        )
        self.model = nn.Sequential(layers)

    def score_emb_spo(self, s_emb, p_emb, o_emb):
        return self.model.forward(torch.cat((s_emb, p_emb, o_emb), 1))


class Fnn(KgeModel):
    r"""Implementation of a simple feedforward neural network KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset, None)
        self._scorer = FnnScorer(
            config,
            dataset,
            self.get_s_embedder().dim,
            self.get_p_embedder().dim,
            self.get_o_embedder().dim,
        )
