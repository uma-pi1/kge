import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp*":
            out = -torch.cdist(s_emb + p_emb, o_emb, p=self._norm)
            # old pytorch 1.2 version
            # sp_emb = s_emb + p_emb
            # out = torch.zeros(n, o_emb.size(0)).to(self.config.get("job.device"))
            # for i in range(n):
            #     out[i, :] = -F.pairwise_distance(sp_emb[i, :], o_emb, p=self._norm)
        elif combine == "*po":
            out = -torch.cdist(o_emb - p_emb, s_emb, p=self._norm)
            # old pytorch 1.2 version
            # po_emb = o_emb - p_emb
            # out = torch.zeros(n, s_emb.size(0)).to(self.config.get("job.device"))
            # for i in range(n):
            #     out[i, :] = -F.pairwise_distance(po_emb[i, :], s_emb, p=self._norm)
        else:
            super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, TransEScorer, configuration_key=configuration_key
        )
