import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel

class SFNNScorer(RelationalScorer):
    r"""Implementation of the Simple feedforward neural network KGE scorer.

    Must be used with InverseRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)
        self.ent_emb_dim = config.get("inverse_relations_model.base_model.entity_embedder.dim")
        self.rel_emb_dim = config.get("inverse_relations_model.base_model.relation_embedder.dim")
        self.hidden_dim = config.get("inverse_relations_model.base_model.hidden_dim")
        self.non_linear = torch.nn.ReLU()

        self.h1 = torch.nn.Linear(self.ent_emb_dim + self.rel_emb_dim, self.hidden_dim)
        self.h1_dropout = torch.nn.Dropout(config.get("inverse_relations_model.base_model.h1_dropout"))
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim, affine=False)

        self.h2 = torch.nn.Linear(self.hidden_dim, int(self.ent_emb_dim))
        self.h2_dropout = torch.nn.Dropout(config.get("inverse_relations_model.base_model.h2_dropout"))
        self.bn2 = torch.nn.BatchNorm1d(self.ent_emb_dim, affine=False)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):

        if combine == "sp*":
            batch_size = p_emb.size(0)
            out = torch.cat([s_emb, p_emb], 1)

            out = self.h1(out)
            out = self.h1_dropout(out)
            out = self.bn1(out)
            out = self.non_linear(out)

            out = self.h2(out)
            out = self.h2_dropout(out)
            out = self.bn2(out)
            out = self.non_linear(out)
            out = torch.mm(out, o_emb.transpose(1, 0))
        else:
            raise Exception("Combine {} not supported in SFNN's score function".format(combine))

        return out.view(batch_size, -1)


class SFNN(KgeModel):
    r"""Implementation of the simple feedforward neural network KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config,
                         dataset,
                         SFNNScorer(config, dataset),
                         configuration_key=configuration_key)
