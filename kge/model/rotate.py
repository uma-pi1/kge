import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


class RotatEScorer(RelationalScorer):
    r"""Implementation of the RotatE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb * p_emb, o_emb, p=self._norm)
        elif combine == "sp*":
            out = -torch.cdist(s_emb * p_emb, o_emb, p=self._norm)
        elif combine == "*po":
            # Predicting all subjects for multiple relations/object pairs is painful
            # (and slow) with RotatE. The reason is that each of the subject embeddings
            # needs to be multiplied with each relation embedding. A simplification as
            # in TransE (where we can "apply" the relation embedding to the objects) is
            # not possible.
            out = torch.zeros(n, s_emb.size(0)).to(self.config.get("job.device"))
            for i in range(n):
                out[i, :] = -torch.cdist(
                     o_emb[i,None], s_emb*p_emb[i,:], p=self._norm
                )
        else:
            super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class RotatE(KgeModel):
    r"""Implementation of the RotatE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, RotatEScorer, configuration_key=configuration_key
        )
