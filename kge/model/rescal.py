import math

import torch

from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer


class RescalScorer(RelationalScorer):
    r"""Implementation of the RESCAL KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(
        self,
        s_emb: torch.Tensor,
        p_emb: torch.Tensor,
        o_emb: torch.Tensor,
        combine: str,
    ):
        batch_size = p_emb.size(0)
        entity_size = s_emb.size(-1)

        # reshape relation embeddings to obtain mixing matrices for RESCAL
        p_mixmat = p_emb.view(-1, entity_size, entity_size)

        if combine == "spo":
            out = (
                s_emb.unsqueeze(1)  # [batch x 1 x entity_size]
                .bmm(p_mixmat)  # apply mixing matrices
                .view(batch_size, entity_size)  # drop dim 1
                * o_emb  # apply object embeddings
            ).sum(
                dim=-1
            )  # and sum to obtain predictions
        elif combine == "sp_":
            out = (
                s_emb.unsqueeze(1)
                .bmm(p_mixmat)
                .view(batch_size, entity_size)
                .mm(o_emb.transpose(0, 1))
            )
        elif combine == "_po":
            out = (
                p_mixmat.bmm(o_emb.unsqueeze(2))
                .view(batch_size, entity_size)
                .mm(s_emb.transpose(0, 1))
            )
        else:
            out = super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(batch_size, -1)


class Rescal(KgeModel):
    r"""Implementation of the RÉSCAL KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        rescal_set_relation_embedder_dim(
            config, dataset, self.configuration_key + ".relation_embedder"
        )
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RescalScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )


def rescal_set_relation_embedder_dim(config, dataset, rel_emb_conf_key):
    """Set the relation embedder dimensionality for RESCAL in the config.

    If <0, set it to the square of the size of the entity embedder. Else leave
    unchanged.

    """
    dim = config.get_default(rel_emb_conf_key + ".dim")
    if dim < 0:  # autodetect relation embedding dimensionality
        ent_emb_conf_key = rel_emb_conf_key.replace(
            "relation_embedder", "entity_embedder"
        )
        if ent_emb_conf_key == rel_emb_conf_key:
            raise ValueError(
                "Cannot determine relation embedding size; please set manually."
            )
        dim = config.get_default(ent_emb_conf_key + ".dim") ** 2
        config.set(rel_emb_conf_key + ".dim", dim, log=True)
