import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        if combine == "spo":
            out = (s_emb * p_emb * o_emb).sum(dim=1)
        elif combine == "sp*":
            out = (s_emb * p_emb).mm(o_emb.transpose(0, 1))
        elif combine == "*po":
            out = (o_emb * o_emb).mm(s_emb.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config,
                         dataset,
                         TransEScorer(config, dataset),
                         configuration_key=configuration_key)
