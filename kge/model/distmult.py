import math

import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class DistMultScorer(RelationalScorer):
    r"""Implementation of the DistMult KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        if combine == "spo":
            out = (s_emb * p_emb * o_emb).sum(dim=1)
        elif combine == "sp_":
            out = (s_emb * p_emb).mm(o_emb.transpose(0, 1))
        elif combine == "_po":
            out = (o_emb * p_emb).mm(s_emb.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class DistMult(KgeModel):
    r"""Implementation of the DistMult KGE model."""

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
            scorer=DistMultScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
