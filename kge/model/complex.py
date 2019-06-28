import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class ComplExScorer(RelationalScorer):
    r"""Implementation of the ComplEx KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # Here we use a fast implementation of computing the ComplEx scores using
        # Hadamard products. TODO add details / reference to paper

        # split the relation and object embeddings into two halves
        p_emb_left, p_emb_right = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_left, o_emb_right = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        # and combine them again
        s_all = torch.cat((s_emb, s_emb), dim=1)
        r_all = torch.cat((p_emb_left, p_emb, -p_emb_right), dim=1)

        o_all = torch.cat((o_emb, o_emb_right, o_emb_left), dim=1)
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
        super().__init__(config,
                         dataset,
                         ComplExScorer(config, dataset),
                         configuration_key=configuration_key)
