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

        # determine real and imaginary party
        s_emb_re, s_emb_im = torch.chunk(s_emb, 2, dim=1)
        o_emb_re, o_emb_im = torch.chunk(s_emb, 2, dim=1)

        # TODO Original RotatE code normalizes relation embeddings here
        p_emb_re, p_emb_im = torch.cos(p_emb), torch.sin(p_emb)

        if combine == "spo":
            # compute the difference vector (s*p-t), treating real and complex parts
            # separately
            sp_emb_re = s_emb_re * p_emb_re - s_emb_im * p_emb_im
            sp_emb_im = s_emb_re * p_emb_im + s_emb_im * p_emb_re
            diff_re = sp_emb_re - o_emb_re
            diff_im = sp_emb_im - o_emb_im

            # compute the absolute values for each (complex) element of the difference
            # vector
            diff = torch.stack((diff_re, diff_im,), dim=0)  # dim0: real, imaginary
            diff_abs = torch.norm(dim=0)  # sqrt(real^2+imaginary^2)

            # now take the norm of the absolute values
            out = torch.norm(diff_abs, dim=1, p=self.norm)
        else:
            super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)


class RotatE(KgeModel):
    r"""Implementation of the RotatE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        if self.get_option("entity_embedder.dim") % 2 != 0:
            raise ValueError(
                "RotatE requires embeddings of even dimensionality"
                " (got {})".format(self.get_option("entity_embedder.dim"))
            )
        if self.get_option("relation_embedder.dim") < 0:
            self.set_option(
                "relation_embedder.dim",
                self.get_option("entity_embedder.dim") // 2,
                log=True,
            )
        super().__init__(
            config, dataset, RotatEScorer, configuration_key=self.configuration_key
        )
