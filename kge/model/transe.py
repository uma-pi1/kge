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
        elif combine == "sp_":
            out = -torch.cdist(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "_po":
            out = -torch.cdist(o_emb - p_emb, s_emb, p=self._norm)
        else:
            out = super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
