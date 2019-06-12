import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, KgeModel, RelationalScorer


class RescalScorer(RelationalScorer):
    r"""Implementation of the RESCAL KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

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
        elif combine == "sp*":
            out = (
                s_emb.unsqueeze(1)
                .bmm(p_mixmat)
                .view(batch_size, entity_size)
                .mm(o_emb.transpose(0, 1))
            )
        elif combine == "*po":
            out = (
                p_mixmat.bmm(o_emb.unsqueeze(2))
                .view(batch_size, entity_size)
                .mm(s_emb.transpose(0, 1))
            )
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(batch_size, -1)


class Rescal(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        rescal_set_relation_embedder_dim(config, dataset, "rescal.relation_embedder")
        super().__init__(
            config, dataset, scorer=RescalScorer(config=config, dataset=dataset)
        )


def rescal_set_relation_embedder_dim(config, dataset, rel_emb_conf_key):
    """Set the relation embedder dimensionality for RESCAL in the config.

    If-1, set it to the square of the size of the entity embedder. Else leave unchanged.

    """
    dim = KgeEmbedder(config, dataset, rel_emb_conf_key).dim
    if dim < 0:  # autodetect relation embedding dimensionality
        entity_dim_key = rel_emb_conf_key.replace(
            "relation_embedder", "entity_embedder"
        )
        if entity_dim_key == rel_emb_conf_key:
            raise ValueError(
                "Cannot determine relation embedding size; please set manually."
            )
        try:
            dim = config.get(entity_dim_key + ".dim") ** 2
        except ValueError:
            dim = config.get(config.get(entity_dim_key + ".type") + ".dim"()) ** 2
        config.set(rel_emb_conf_key + ".dim", dim)
