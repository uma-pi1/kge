import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as f


class TransRScorer(RelationalScorer):
    r"""Implementation of the TransR KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def _transfer(self, ent_emb, rel_norm_vec_emb):
        k = rel_norm_vec_emb.size(1)
        d = ent_emb.size(1)
        rel_hat = rel_norm_vec_emb.divide(torch.norm(rel_norm_vec_emb))

        return (ent_emb.view(d, -1).matmul(rel_hat.view(-1, k)).matmul(rel_hat.view(k, -1))).view(-1, d)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        rel_emb, rel_norm_vec_emb = torch.chunk(p_emb, 2, dim=1)

        n = p_emb.size(0)
        if combine == "spo":
            # n = n_s = n_p = n_o
            s_emb_translated = self._transfer(s_emb, rel_norm_vec_emb)
            o_emb_translated = self._transfer(o_emb, rel_norm_vec_emb)

            out = -f.pairwise_distance(s_emb_translated + p_emb, o_emb_translated, p=self._norm)
        elif combine == "sp_":
            # n = n_s = n_p != n_o = m
            m = o_emb.size(0)

            s_emb_translated = self._transfer(s_emb, rel_norm_vec_emb)
            s_emb_translated = s_emb_translated.repeat(m, 1)

            p_emb = p_emb.repeat(m, 1)
            rel_norm_vec_emb = rel_norm_vec_emb.repeat(m, 1)

            o_emb = o_emb.repeat_interleave(n, 0)
            o_emb_translated = self._transfer(o_emb, rel_norm_vec_emb)

            out = -f.pairwise_distance(s_emb_translated + p_emb, o_emb_translated, p=self._norm)
            out = out.view(m, n).transpose(0, 1)
        elif combine == "_po":
            # m = n_s != n_p = n_o = n
            m = s_emb.size(0)

            o_emb_translated = self._transfer(o_emb, rel_norm_vec_emb)
            o_emb_translated = o_emb_translated.repeat(m, 1)

            p_emb = p_emb.repeat(m, 1)
            rel_norm_vec_emb = rel_norm_vec_emb.repeat(m, 1)

            s_emb = s_emb.repeat_interleave(n, 0)
            s_emb_translated = self._transfer(s_emb, rel_norm_vec_emb)

            out = -f.pairwise_distance(o_emb_translated - p_emb, s_emb_translated, p=self._norm)
            out = out.view(m, n).transpose(0, 1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class TransR(KgeModel):
    r"""Implementation of the TransR KGE model."""

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
            scorer=TransRScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
