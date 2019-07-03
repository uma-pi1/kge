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
            if combine == "spo":
                out = torch.norm(s_emb + p_emb - o_emb,
                                 p=self._norm,
                                 dim=1) * -1
            elif combine == "sp*":
                out = torch.zeros(n, o_emb.size(0)).to(self.config.get("job.device"))
                for i in range(n):
                    out[i,:] = torch.norm((s_emb[i,:] + p_emb[i,:]) - o_emb,
                                          p=self._norm,
                                          dim=1) * -1
            elif combine == "*po":
                out = torch.zeros(n, s_emb.size(0)).to(self.config.get("job.device"))
                for i in range(n):
                    out[i,:] = torch.norm(s_emb + (p_emb[i,:] - o_emb[i,:]),
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
