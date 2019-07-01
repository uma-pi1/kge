import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)
        self._norm = config.get("transe.l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        emb_dim = p_emb.size(1)
        if combine == "spo":
            out = torch.norm(s_emb + p_emb - o_emb,
                             p=self._norm,
                             dim=1) * -1
        elif combine == "sp*":
            s_emb = s_emb.repeat(1, o_emb.size(0)).view(-1, emb_dim)
            p_emb = p_emb.repeat(1, o_emb.size(0)).view(-1, emb_dim)
            o_emb = o_emb.repeat(n, 1)
            out = torch.norm(s_emb + p_emb - o_emb,
                             p=self._norm,
                             dim=1) * -1
        elif combine == "*po":
            p_emb = p_emb.repeat(1, s_emb.size(0)).view(-1, emb_dim)
            o_emb = o_emb.repeat(1, s_emb.size(0)).view(-1, emb_dim)
            s_emb = s_emb.repeat(n, 1)
            out = torch.norm(s_emb + p_emb - o_emb,
                             p=self._norm,
                             dim=1) * -1
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
