import torch
import torch.nn as nn
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.misc import get_activation_function
from collections import OrderedDict


class FnnScorer(RelationalScorer):
    r"""Implementation of a simple feedforward neural network KGE scorer.

    Concatenates the spo embeddings and runs them through a fully connected neural
    network with a linear output unit. The number of hidden layers as well as their
    sizes can be configured.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        subject_dim: int,
        relation_dim: int,
        object_dim: int,
        configuration_key=None,
    ):
        super().__init__(config, dataset, configuration_key)

        hidden_layers_size = self.get_option("hidden_layers.size")
        if hidden_layers_size < 0:
            hidden_layers_size = subject_dim
        hidden_layers_dropout = self.get_option("hidden_layers.dropout")
        layers = OrderedDict()
        last_size = subject_dim + relation_dim + object_dim
        for i in range(self.get_option("hidden_layers.number")):  # hidden layers
            layer = nn.Linear(last_size, hidden_layers_size)
            last_size = hidden_layers_size
            self.initialize(
                layer.weight,
                self.get_option("hidden_layers.initialize"),
                self.get_option("hidden_layers.initialize_args"),
            )
            layers["linear" + str(i + 1)] = layer
            layers["nonlinear" + str(i + 1)] = get_activation_function(
                self.get_option("hidden_layers.activation")
            )
            if (
                hidden_layers_dropout > 0.0
            ):  # note: input hidden_layers_dropout handled separately by embedder
                layers["hidden_layers_dropout" + str(i + 1)] = nn.Dropout(
                    hidden_layers_dropout
                )

        layers["output"] = nn.Linear(last_size, 1)
        self.initialize(
            layers["output"].weight,
            self.get_option("hidden_layers.initialize"),
            self.get_option("hidden_layers.initialize_args"),
        )
        self.model = nn.Sequential(layers)

    def score_emb_spo(self, s_emb, p_emb, o_emb):
        return self.model.forward(torch.cat((s_emb, p_emb, o_emb), 1))


class Fnn(KgeModel):
    r"""Implementation of a simple feedforward neural network KGE model.

    Note that this model is very slow since no computation can be shared when scoring
    related triples (e.g., as arise when scoring all objects for an (s,p,?) task).

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, scorer=None, configuration_key=configuration_key
        )
        self._scorer = FnnScorer(
            config,
            dataset,
            self.get_s_embedder().dim,
            self.get_p_embedder().dim,
            self.get_o_embedder().dim,
            configuration_key=self.configuration_key,
        )
