import math

import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class CPScorer(RelationalScorer):
    r"""Implementation of the CP KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # use only first half for subjects and second half for objects
        s_emb_l = s_emb[:, : (s_emb.shape[1] // 2)]
        o_emb_r = o_emb[:, (o_emb.shape[1] // 2) :]

        if combine == "spo":
            out = (s_emb_l * p_emb * o_emb_r).sum(dim=1)
        elif combine == "sp*":
            out = (s_emb_l * p_emb).mm(o_emb_r.transpose(0, 1))
        elif combine == "*po":
            out = (o_emb_r * p_emb).mm(s_emb_l.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class CP(KgeModel):
    r"""Implementation of the CP KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        if self.get_option("entity_embedder.dim") % 2 != 0:
            raise ValueError(
                "CP requires embeddings of even dimensionality"
                " (got {})".format(self.get_option("entity_embedder.dim"))
            )
        if self.get_option("relation_embedder.dim") < 0:
            self.set_option(
                "relation_embedder.dim",
                self.get_option("entity_embedder.dim") // 2,
                log=True,
            )
        super().__init__(
            config, dataset, CPScorer, configuration_key=self.configuration_key
        )
