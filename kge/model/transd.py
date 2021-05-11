import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F
from torch import Tensor
from typing import List


class TransDScorer(RelationalScorer):
    r"""Implementation of the TransD KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    @staticmethod
    def _transfer(ent_emb, ent_norm_vec_emb, rel_norm_vec_emb):
        d = ent_norm_vec_emb.size(1)
        k = rel_norm_vec_emb.size(1)

        min_dim = min(d, k)

        out = rel_norm_vec_emb * torch.sum(
            ent_norm_vec_emb * ent_emb, dim=-1, keepdim=True
        )
        out[:, :min_dim] += ent_emb[:, :min_dim]

        return out

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        # split relation embeddings into "rel_emb" and "norm_vec_emb"
        sub_emb, sub_norm_vec_emb = torch.chunk(s_emb, 2, dim=1)
        rel_emb, rel_norm_vec_emb = torch.chunk(p_emb, 2, dim=1)
        obj_emb, obj_norm_vec_emb = torch.chunk(o_emb, 2, dim=1)

        # TODO sp_ and _po is currently very memory intensive since each _ must be
        # projected once for every different relation p. Unclear if this can be avoided.

        n = p_emb.size(0)
        if combine == "spo":
            # n = n_s = n_p = n_o
            out = -F.pairwise_distance(
                self._transfer(sub_emb, sub_norm_vec_emb, rel_norm_vec_emb) + rel_emb,
                self._transfer(obj_emb, obj_norm_vec_emb, rel_norm_vec_emb),
                p=self._norm,
            )
        elif combine == "sp_":
            # n = n_s = n_p != n_o = m
            m = o_emb.shape[0]
            s_translated = (
                self._transfer(sub_emb, sub_norm_vec_emb, rel_norm_vec_emb) + rel_emb
            )
            s_translated = s_translated.repeat(m, 1)
            # s_translated has shape [(m * n), dim d]
            rel_norm_vec_emb = rel_norm_vec_emb.repeat(m, 1)
            # rel_norm_vec_emb has shape [(m * n), dim k]
            obj_emb = obj_emb.repeat_interleave(n, 0)
            obj_norm_vec_emb = obj_norm_vec_emb.repeat_interleave(n, 0)
            # obj_emb and obj_norm_vec_emb have shape [(m * n), dim d]
            o_translated = self._transfer(obj_emb, obj_norm_vec_emb, rel_norm_vec_emb)
            # o_translated has shape [(m * n), dim d]
            # --> perform pairwise distance calculation
            out = -F.pairwise_distance(s_translated, o_translated, p=self._norm)
            # out has shape [(m * n)]
            # --> transform shape to [n, m]
            out = out.view(m, n)
            out = out.transpose(0, 1)
        elif combine == "_po":
            # m = n_s != n_p = n_o = n
            m = s_emb.shape[0]
            o_translated = (
                self._transfer(obj_emb, obj_norm_vec_emb, rel_norm_vec_emb) - rel_emb
            )
            o_translated = o_translated.repeat(m, 1)
            # o_translated has shape [(m * n), dim]
            rel_norm_vec_emb = rel_norm_vec_emb.repeat(m, 1)
            # rel_norm_vec_emb has shape [(m * n), dim k]
            sub_emb = sub_emb.repeat_interleave(n, 0)
            sub_norm_vec_emb = sub_norm_vec_emb.repeat_interleave(n, 0)
            # sub_emb and sub_norm_vec_emb have shape [(m * n), dim d]
            s_translated = self._transfer(sub_emb, sub_norm_vec_emb, rel_norm_vec_emb)
            # s_translated has shape [(m * n), dim]
            # --> perform pairwise distance calculation
            out = -F.pairwise_distance(o_translated, s_translated, p=self._norm)
            # out has shape [(m * n)]
            # --> transform shape to [n, m]
            out = out.view(m, n)
            out = out.transpose(0, 1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class TransD(KgeModel):
    r"""Implementation of the TransD KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransDScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
