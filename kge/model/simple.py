import math

import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class SimplEScorer(RelationalScorer):
    r"""Implementation of the SimplE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # split left/right
        half_dim = s_emb.shape[1] // 2
        s_emb_h = s_emb[:, :half_dim]
        p_emb_forward = p_emb[:, :half_dim]
        o_emb_h = o_emb[:, :half_dim]
        s_emb_t = s_emb[:, half_dim:]
        p_emb_backward = p_emb[:, half_dim:]
        o_emb_t = o_emb[:, half_dim:]

        if combine == "spo":
            out1 = (s_emb_h * p_emb_forward * o_emb_t).sum(dim=1)
            out2 = (s_emb_t * p_emb_backward * o_emb_h).sum(dim=1)
        elif combine == "sp*":
            out1 = (s_emb_h * p_emb_forward).mm(o_emb_t.transpose(0, 1))
            out2 = (s_emb_t * p_emb_backward).mm(o_emb_h.transpose(0, 1))
        elif combine == "*po":
            out1 = (o_emb_t * p_emb_forward).mm(s_emb_h.transpose(0, 1))
            out2 = (o_emb_h * p_emb_backward).mm(s_emb_t.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return (out1 + out2).view(n, -1) / 2.0


class SimplE(KgeModel):
    r"""Implementation of the SimplE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        if self.get_option("entity_embedder.dim") % 2 != 0:
            raise ValueError(
                "SimplE requires embeddings of even dimensionality"
                " (got {})".format(self.get_option("entity_embedder.dim"))
            )
        super().__init__(
            config, dataset, SimplEScorer, configuration_key=self.configuration_key
        )
