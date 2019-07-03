import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)
        self._norm = config.get("transe.l_norm")

    def score_emb_spo(self, s_emb, p_emb, o_emb):
        n = p_emb.size(0)
        out = torch.norm(s_emb + p_emb - o_emb,
                         p=self._norm,
                         dim=1) * -1
        return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config,
                         dataset,
                         TransEScorer(config, dataset),
                         configuration_key=configuration_key)
