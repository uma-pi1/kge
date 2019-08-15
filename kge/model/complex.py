import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class ComplExScorer(RelationalScorer):
    r"""Implementation of the ComplEx KGE scorer.

    Reference: Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and
    Guillaume Bouchard: Complex Embeddings for Simple Link Prediction. ICML 2016.
    `<http://proceedings.mlr.press/v48/trouillon16.pdf>`_

    """

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # Here we use a fast implementation of computing the ComplEx scores using
        # Hadamard products, as in Eq. (11) of paper.
        #
        # Split the relation and object embeddings into real part (first half) and
        # imaginary part (second half).
        p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        # combine them again to create a column block for each required combination
        s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
        r_all = torch.cat((p_emb_re, p_emb, -p_emb_im), dim=1)  # re, re, im, -im
        o_all = torch.cat((o_emb, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

        if combine == "spo":
            out = (s_all * o_all * r_all).sum(dim=1)
        elif combine == "sp*":
            out = (s_all * r_all).mm(o_all.transpose(0, 1))
        elif combine == "*po":
            out = (r_all * o_all).mm(s_all.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class ComplEx(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # auto initialize such that scores have unit variance
        if self.get_option("entity_embedder.initialize") == "auto_initialization" and \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            # Var[score] = 4*(dim/2)*var_e^2*var_r, where var_e/var_r are the variances
            # of the entries in the embeddings and (dim/2) is the embedding size in
            # complex space
            #
            # Thus we set var_e=var_r=(1.0/(2.0*dim))^(1/3)
            dim = self.get_option("entity_embedder.dim")
            std = math.pow(1.0 / 2.0 / dim, 1.0 / 6.0)

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

        super().__init__(
            config,
            dataset,
            ComplExScorer(config, dataset),
            configuration_key=configuration_key,
        )
