import math

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
    r"""Implementation of the RÉSCAL KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        rescal_set_relation_embedder_dim(config, dataset, self.configuration_key + ".relation_embedder")

        # auto initialize such that scores have unit variance
        if self.get_option("entity_embedder.initialize") == "auto_initialization" and \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            # Var[score] = entity_embedder.dim^2*var_e^2*var_r, where var_e/var_r are the variances
            # of the entries
            #
            # Thus we set var_e=var_r=(1.0/(entity_embedder.dim^2*))^(1/6)
            std = math.pow(1.0 / self.get_option("entity_embedder.dim") ** 2, 1.0 / 6.0)

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
            scorer=RescalScorer(config=config, dataset=dataset),
            configuration_key=configuration_key
        )



def rescal_set_relation_embedder_dim(config, dataset, rel_emb_conf_key):
    """Set the relation embedder dimensionality for RESCAL in the config.

    If -1, set it to the square of the size of the entity embedder. Else leave unchanged.

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
