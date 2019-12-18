import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class FreexScorer(RelationalScorer):
    """Mixing matrix with ComplEx pattern of entries, but entries free.

    Initial sandbox model for experimentation with sparsity patterns.
    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        o_emb1, o_emb2 = (t.contiguous() for t in o_emb.chunk(2, dim=1))
        s_all = torch.cat((s_emb, s_emb), dim=1)
        o_all = torch.cat((o_emb, o_emb2, o_emb1), dim=1)

        if combine == "spo":
            out = (s_all * o_all * p_emb).sum(dim=1)
        elif combine == "sp*":
            out = (s_all * p_emb).mm(o_all.transpose(0, 1))
        elif combine == "*po":
            out = (p_emb * o_all).mm(s_all.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class Freex(KgeModel):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # set relation embedder dimensionality
        if self.get_option("relation_embedder.dim") < 0:
            self.config.set(
                self.configuration_key + ".relation_embedder.dim",
                2 * self.get_option("entity_embedder.dim"),
                log=True,
            )

        super().__init__(
            config, dataset, FreexScorer, configuration_key=self.configuration_key
        )
