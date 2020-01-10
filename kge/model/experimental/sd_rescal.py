import math
from typing import List

import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, RelationalScorer
from kge.misc import round_to_points


class SparseDiagonalRescalScorer(RelationalScorer):
    r"""Implementation of the Sparse Diagonal RESCAL KGE scorer."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        blocks,
        block_size,
        configuration_key=None,
    ):
        super().__init__(config, dataset, configuration_key)
        self.blocks = blocks
        self.block_size = block_size

    def score_emb(
        self,
        s_emb: torch.Tensor,
        p_emb: torch.Tensor,
        o_emb: torch.Tensor,
        combine: str,
    ):
        batch_size = p_emb.size(0)
        entity_size = s_emb.size(-1)

        #   We explain the implementation by example as follows:
        #
        # 	blocks 2
        # 	block size 2
        # 	 →	entity embedding size 4
        # 	 →	relation embedding size 8
        #
        #   Relation embedding:
        #
        # 	M =	[m1	m2	m3	m4	m5	m6	m7	m8	]
        #
        #   Entity embeddings (called left and right for the left and right hand side
        #   of the bilinear product):
        #
        # 	l(eft)  = 	[	l1	l2	l3	l4	]
        # 	r(ight) = 	[	r1	r2	r3	r4	]
        #
        #
        #   Computing the bilinear product for *po:
        #   ---------------------------------------
        #
        # 	l^T x M x r
        #
        # 			r1	r2	r3	r4
        #
        # 	l1		m1		m3
        # 	l2			m2		m4
        # 	l3		m5		m7
        # 	l4			m6		m8
        #
        #   We compute M x r with the following
        #
        # 	M * r   =   m1	m2	m3	m4	m5	m6	m7	m8  *   r1	r2	r3	r4	r1	r2	r3	r4
        #
        # 	View(1,2,2,2) →	m1*r1		m2*r2
        # 					m3*r3		m4*r4
        #
        # 					m5*r1		m6*r2
        # 					m7*r3		m8*r4
        #
        # 	Sum(-2)			m1*r1+m3*r3
        # 					m2*r2+m4*r4
        #
        # 					m5*r1+m7*r3
        # 					m6*r2+m8*r4
        #
        #
        #   l^T x M x r =
        #   ( M * r ).View(1,2,2,2).Sum(-2) x l^T
        #
        #
        #   Computing the bilinear product for sp*:
        #   ---------------------------------------
        #
        # 	l x M^T x r^T
        #
        #   first transpose M and flatten it to compute l x M^T
        #
        # 	M.View(1,2,2,2) →	m1		m2
        # 						m3		m4
        #
        # 						m5		m6
        # 						m7		m8
        #
        # 	Permute(0,2,1,3) →	m1		m2
        # 						m5		m6
        #
        # 						m3		m4
        # 						m7		m8
        #
        # 	View(1,-1) →	m1	m2	m5	m6	m3	m4	m7	m8
        #
        # 			l1	l2	l3	l4
        #
        # 	r1		m1		m5
        # 	r2			m2		m6
        # 	r3		m3		m7
        # 	r4			m4		m8
        #
        #   now we can compute l x M^T
        #
        #  ( l * M^T ).View(1,2,2,2).Sum(-2) x r^T

        if combine in ["*po", "sp*"]:

            if combine == "sp*":
                p_emb = (
                    p_emb.view(batch_size, self.blocks, self.blocks, self.block_size)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(batch_size, -1)
                )
                left = o_emb
                right = s_emb

            if combine == "*po":
                left = s_emb
                right = o_emb

            right_repeated = right.repeat(1, 1, self.blocks).view(
                batch_size, entity_size * self.blocks
            )
            p_r_strided = (
                (p_emb * right_repeated)
                .view(batch_size, self.blocks, self.blocks, self.block_size)
                .sum(-2)
                .view(batch_size, -1)
            )

            out = p_r_strided.mm(left.transpose(0, 1))

        elif combine == "spo":
            raise ValueError('cannot handle combine="{}".format(combine)')
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(batch_size, -1)


class SparseDiagonalRescal(KgeModel):
    r"""Implementation of the Sparse Diagonal RESCAL KGE model which includes
    Distmult, unconstrained Complex or RECSAL.

    This implementation of Sparse Diagonal Rescal is as fast as Complex,
    Distmult or RESCAL. The model is defined by the number of blocks and the
    block size. To define the model two out of the following hyper-parameters
    have to be defined in the config: blocks, block_size and entity embedding
    size; the third undefined parameter and relation embedding size will be
    inferred automatically.

    Here is an example that yields unconstrained ComplEx:

    blocks 2
    block size 2
     →	entity embedding size 4
     →	relation embedding size 8

    Relation embedding:

    M =	[	m1	m2	m3	m4	m5	m6	m7	m8	]

    e1  = 	[	e11	e12	e13	e14	]
    e2 = 	[	e21	e22	e23	e24	]

    Computing the bilinear product:

    e1^T x M x e2

            e21	e22	e23	e24

    e11		m1		m3
    e12			m2		m4
    e13		m5		m7
    e14			m6		m8

    Sparse Diagonal RESCAL contains: Distmult with blocks = 1 and
    block size = entity size, unconstrained ComplEx with blocks = 2 and
    block size is half of entity size RESCAL is blocks = entity size and
    block size is 1. See the score_emb function for details about the
    implementation.
    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        rel_emb_conf_key = self.configuration_key + ".relation_embedder"
        ent_emb_conf_key = self.configuration_key + ".entity_embedder"

        round_blocks_to = config.get_default(self.configuration_key + ".round_blocks_to")
        round_block_size_to = config.get_default(self.configuration_key + ".round_block_size_to")
        round_ent_emb_dim_to = config.get_default(ent_emb_conf_key + ".round_dim_to")

        blocks = config.get_default(self.configuration_key + ".blocks")
        block_size = config.get_default(self.configuration_key + ".block_size")

        rel_emb_dim = config.get_default(rel_emb_conf_key + ".dim")
        ent_emb_dim = config.get_default(ent_emb_conf_key + ".dim")

        if len(round_blocks_to) > 0:
            blocks = round_to_points(round_blocks_to, blocks)

        if len(round_block_size_to) > 0:
            block_size = round_to_points(round_block_size_to, block_size)

        if len(round_ent_emb_dim_to) > 0:
            ent_emb_dim = round_to_points(round_ent_emb_dim_to, ent_emb_dim)

        if rel_emb_dim > 0:
            raise ValueError(
                "Relation embedding sizes are determined automatically from "
                "sd_rescal.blocks and sd_rescal.block_size or entity_embedding.dim; "
                "do not set manually."
            )

        if blocks <= 0 and block_size > 0 and ent_emb_dim > 0:
            if ent_emb_dim % block_size != 0:
                raise ValueError(
                    "If blocks <= 0 then block_size ({}) and entity_size ({}) have "
                    "to be larger than 0 and entity_size has to be dividable by "
                    "block_size".format(block_size, ent_emb_dim)
                )
            blocks = ent_emb_dim // block_size

        if block_size <= 0 and blocks > 0 and ent_emb_dim > 0:
            if ent_emb_dim % blocks != 0:
                raise ValueError(
                    "If block_size <= 0 then blocks ({}) and entity_size ({}) have "
                    "to be larger than 0 and entity_size has to be dividable by "
                    "blocks".format(block_size, ent_emb_dim)
                )
            block_size = ent_emb_dim // blocks

        config.set(ent_emb_conf_key + ".dim", blocks * block_size, log=True)
        config.set(rel_emb_conf_key + ".dim", blocks ** 2 * block_size, log=True)
        config.set(self.configuration_key + ".blocks", blocks, log=True)
        config.set(self.configuration_key + ".block_size", block_size, log=True)

        # TODO remove auto_init
        # auto initialize such that scores have unit variance

        if self.get_option("relation_embedder.type") == "projection_embedder":
            relation_embedder = "relation_embedder.base_embedder"
        else:
            relation_embedder = "relation_embedder"

        if (
            self.get_option("entity_embedder.initialize") == "auto_initialization"
            and self.get_option(relation_embedder + ".initialize")
            == "auto_initialization"
        ):
            # Var[score] = blocks^2*block_size*var_e^2*var_r, where var_e/var_r are the variances
            # of the entries
            #
            # Thus we set var_e=var_r=(1.0/(blocks^2*block_size))^(1/6)
            std = math.pow(1.0 / (blocks ** 2 * block_size), 1.0 / 6.0)

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

            if (
                config.get_default(self.configuration_key + ".relation_embedder.type")
                == "projection_embedder"
            ):
                # core tensor weight -> initial scores have var=1 (when no dropout / eval)
                config.set(
                    self.configuration_key + ".relation_embedder.initialize",
                    "normal_",
                    log=True,
                )
                config.set(
                    self.configuration_key + ".relation_embedder.initialize_args",
                    {"mean": 0.0, "std": 1.0},
                    log=True,
                )

            config.set(
                self.configuration_key + "." + relation_embedder + ".initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + "." + relation_embedder + ".initialize_args",
                {"mean": 0.0, "std": std},
                log=True,
            )

        elif (
            self.get_option("entity_embedder.initialize") == "auto_initialization"
            or self.get_option(relation_embedder + ".initialize")
            == "auto_initialization"
        ):
            raise ValueError(
                "Both entity and relation embedders must be set to auto_initialization "
                "in order to use it."
            )

        super().__init__(
            config,
            dataset,
            scorer=SparseDiagonalRescalScorer(
                config=config, dataset=dataset, blocks=blocks, block_size=block_size
            ),
            configuration_key=self.configuration_key,
        )
        config.set(rel_emb_conf_key + ".dim", 0)
