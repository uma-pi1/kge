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
        half_dim = s_emb.shape[1] // 2
        s_emb_h = s_emb[:, :half_dim]
        o_emb_t = o_emb[:, half_dim:]

        if combine == "spo":
            out = (s_emb_h * p_emb * o_emb_t).sum(dim=1)
        elif combine == "sp_":
            out = (s_emb_h * p_emb).mm(o_emb_t.transpose(0, 1))
        elif combine == "_po":
            out = (o_emb_t * p_emb).mm(s_emb_h.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class CP(KgeModel):
    r"""Implementation of the CP KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
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
            config=config,
            dataset=dataset,
            scorer=CPScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
