import math

import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class DistMultScorer(RelationalScorer):
    r"""Implementation of the DistMult KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        if combine == "spo":
            out = (s_emb * p_emb * o_emb).sum(dim=1)
        elif combine == "sp*":
            out = (s_emb * p_emb).mm(o_emb.transpose(0, 1))
        elif combine == "*po":
            out = (o_emb * p_emb).mm(s_emb.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class DistMult(KgeModel):
    r"""Implementation of the DistMult KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        # auto initialize such that scores have unit variance
        if self.get_option("entity_embedder.initialize") == "auto_initialization" and \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            # Var[score] = entity_embedder.dim*var_e^2*var_r, where var_e/var_r are the variances
            # of the entries
            #
            # Thus we set var_e=var_r=(1.0/(entity_embedder.dim))^(1/6)
            std = math.pow(1.0 / self.get_option("entity_embedder.dim"), 1.0 / 6.0)

            config.set(
                self.configuration_key + ".entity_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".entity_embedder.initialize_args",
                {"mean": 0.0, "std": std},
                log=True,
            )
            config.set(
                self.configuration_key + ".relation_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".relation_embedder.initialize_args",
                {"mean": 0.0, "std": std},
                log=True,
            )
        elif self.get_option("entity_embedder.initialize") == "auto_initialization" or \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            raise ValueError("Both entity and relation embedders must be set to auto_initialization "
                             "in order to use it.")

        super().__init__(config,
                         dataset,
                         DistMultScorer(config, dataset),
                         configuration_key=configuration_key)
