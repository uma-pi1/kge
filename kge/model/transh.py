import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F
from torch import Tensor
from typing import List


class TransHScorer(RelationalScorer):
    r"""Implementation of the TransH KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    @staticmethod
    def _transfer(ent_emb, norm_vec_emb):
        norm_vec_emb = F.normalize(norm_vec_emb, p=2, dim=-1)
        return (
            ent_emb
            - torch.sum(ent_emb * norm_vec_emb, dim=-1, keepdim=True) * norm_vec_emb
        )

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        # split relation embeddings into "rel_emb" and "norm_vec_emb"
        rel_emb, norm_vec_emb = torch.chunk(p_emb, 2, dim=1)

        # TODO sp_ and _po is currently very memory intensive since each _ must be
        # projected once for every different relation p. Unclear if this can be avoided.

        n = p_emb.size(0)
        if combine == "spo":
            # n = n_s = n_p = n_o
            out = -F.pairwise_distance(
                self._transfer(s_emb, norm_vec_emb) + rel_emb,
                self._transfer(o_emb, norm_vec_emb),
                p=self._norm,
            )
        elif combine == "sp_":
            # n = n_s = n_p != n_o = m
            m = o_emb.shape[0]
            s_translated = self._transfer(s_emb, norm_vec_emb) + rel_emb
            s_translated = s_translated.repeat(m, 1)
            # s_translated has shape [(m * n), dim]
            o_emb = o_emb.unsqueeze(1)
            o_emb = o_emb.repeat(1, n, 1)
            # o_emb has shape [m, n, dim]
            # norm_vec_emb has shape [n, dim]
            # --> make use of broadcasting semantics
            o_emb = self._transfer(o_emb, norm_vec_emb)
            o_emb = o_emb.view(-1, o_emb.shape[-1])
            # o_emb has shape [(m * n), dim]
            # --> perform pairwise distance calculation
            out = -F.pairwise_distance(s_translated, o_emb, p=self._norm)
            # out has shape [(m * n)]
            # --> transform shape to [n, m]
            out = out.view(m, n)
            out = out.transpose(0, 1)
        elif combine == "_po":
            # m = n_s != n_p = n_o = n
            m = s_emb.shape[0]
            o_translated = self._transfer(o_emb, norm_vec_emb) - rel_emb
            o_translated = o_translated.repeat(m, 1)
            # o_translated has shape [(m * n), dim]
            s_emb = s_emb.unsqueeze(1)
            s_emb = s_emb.repeat(1, n, 1)
            # s_emb has shape [m, n, dim]
            # norm_vec_emb has shape [n, dim]
            # --> make use of broadcasting semantics
            s_emb = self._transfer(s_emb, norm_vec_emb)
            s_emb = s_emb.view(-1, s_emb.shape[-1])
            # s_emb has shape [(m * n), dim]
            # --> perform pairwise distance calculation
            out = -F.pairwise_distance(o_translated, s_emb, p=self._norm)
            # out has shape [(m * n)]
            # --> transform shape to [n, m]
            out = out.view(m, n)
            out = out.transpose(0, 1)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class TransH(KgeModel):
    r"""Implementation of the TransH KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        transh_set_relation_embedder_dim(
            config, dataset, self.configuration_key + ".relation_embedder"
        )
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=TransHScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.soft_constraint_weight = self.get_option("C")

    def penalty(self, **kwargs) -> List[Tensor]:
        penalty_super = super().penalty(**kwargs)

        if self.soft_constraint_weight > 0.:
            # entity penalty
            p_ent = F.relu(
                torch.norm(self._entity_embedder.embed_all(), dim=1) ** 2.0 - 1.0
            ).sum()

            # relation penalty
            rel_emb, norm_vec_emb = torch.chunk(
                self._relation_embedder.embed_all(), 2, dim=1
            )

            # NOTE PR #176: added "+ eps" to denominator to prevent blow-up due to
            # division by very small numbers
            eps = 1e-6  # paper is silent on how to set this
            p_rel = torch.sum(
                F.relu(
                    (
                        torch.sum(rel_emb * norm_vec_emb, dim=-1)
                        / (torch.norm(rel_emb, dim=1) + eps)
                    )
                    ** 2
                    - eps ** 2
                )
            )

            return (
                penalty_super
                + [("transh.soft_constraints_ent", self.soft_constraint_weight * p_ent)]
                + [("transh.soft_constraints_rel", self.soft_constraint_weight * p_rel)]
            )
        else:
            return penalty_super


def transh_set_relation_embedder_dim(config, dataset, rel_emb_conf_key):
    """Set the relation embedder dimensionality for TransH in the config.

    Dimensionality must be double the size of the entity embedder dimensionality.

    """
    dim = config.get_default(rel_emb_conf_key + ".dim")
    if dim < 0:
        ent_emb_conf_key = rel_emb_conf_key.replace(
            "relation_embedder", "entity_embedder"
        )
        if ent_emb_conf_key == rel_emb_conf_key:
            raise ValueError(
                "Cannot determine relation embedding size. "
                "Please set manually to double the size of the "
                "entity embedder dimensionality."
            )
        dim = config.get_default(ent_emb_conf_key + ".dim") * 2
        config.set(rel_emb_conf_key + ".dim", dim, log=True)
