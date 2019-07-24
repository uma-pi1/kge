import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, RelationalScorer


class SparseDiagonalRescalScorer(RelationalScorer):
    r"""Implementation of the Sparse Diagonal RESCAL KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, blocks, block_size):
        super().__init__(config, dataset)
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
        #	blocks 2
        #	block size 2
        #	 →	entity embedding size 4
        #	 →	relation embedding size 8
        #
        #   Relation embedding:
        #
        #	M =	[	M1	M2	M3	M4	M5	M6	M7	M8	]
        #
        #   Entity embeddings (called left and right for the left and right hand side
        #   of the bilinear product):
        #
        #	l(eft)  = 	[	l1	l2	l3	l4	]
        #	r(ight) = 	[	r1	r2	r3	r4	]
        #
        #
        #   Computing the bilinear product for *po:
        #   ---------------------------------------
        #
        #	l^T x M x r
        #
        #			r1	r2	r3	r4
        #
        #	l1		M1		M3
        #	l2			M2		M4
        #	l3		M5		M7
        #	l4			M6		M8
        #
        #   We compute M x r with the following
        #
        #	M * r   =   M1	M2	M3	M4	M5	M6	M7	M8  *   r1	r2	r3	r4	r1	r2	r3	r4
        #
        #	View(1,2,2,2) →	M1*r1		M2*r2
        #					M3*r3		M4*r4
        #
        #					M5*r1		M6*r2
        #					M7*r3		M8*r4
        #
        #	Sum(-2)			M1*r1+M3*r3
        #					M2*r2+M4*r4
        #
        #					M5*r1+M7*r3
        #					M6*r2+M8*r4
        #
        #
        #   l^T x M x r =
        #   ( M * r ).View(1,2,2,2).Sum(-2) x l^T
        #
        #
        #   Computing the bilinear product for sp*:
        #   ---------------------------------------
        #
        #	l x M^T x r^T
        #
        #   first transpose M and flatten it to compute l x M^T
        #
        #	M.View(1,2,2,2) →	M1		M2
        #						M3		M4
        #
        #						M5		M6
        #						M7		M8
        #
        #	Permute(0,2,1,3) →	M1		M2
        #						M5		M6
        #
        #						M3		M4
        #						M7		M8
        #
        #	View(1,-1) →	M1	M2	M5	M6	M3	M4	M7	M8
        #
        #			l1	l2	l3	l4
        #
        #	r1		M1		M5
        #	r2			M2		M6
        #	r3		M3		M7
        #	r4			M4		M8
        #
        #   now we can compute l x M^T
        #
        #  ( l * M^T ).View(1,2,2,2).Sum(-2) x r^T

        if combine in ["*po", "sp*"]:

            if combine == "sp*":
                p_emb = p_emb.\
                    view(batch_size, self.blocks, self.blocks, self.block_size).\
                    permute(0, 2, 1, 3).contiguous().\
                    view(batch_size, -1)
                left = o_emb
                right = s_emb

            if combine == "*po":
                left = s_emb
                right = o_emb

            right_repeated = right.repeat(1, 1, self.blocks).\
                view(batch_size, entity_size * self.blocks)
            p_r_strided = (p_emb*right_repeated).\
                view(batch_size, self.blocks, self.blocks, self.block_size).\
                sum(-2).\
                view(batch_size,-1)

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

    M =	[	M1	M2	M3	M4	M5	M6	M7	M8	]

    e1  = 	[	e11	e12	e13	e14	]
    e2 = 	[	e21	e22	e23	e24	]

    Computing the bilinear product:

    e1^T x M x e2

            e21	e22	e23	e24

    e11		M1		M3
    e12			M2		M4
    e13		M5		M7
    e14			M6		M8

    Sparse Diagonal RESCAL contains: Distmult with blocks = 1 and
    block size = entity size, unconstrained ComplEx with blocks = 2 and
    block size is half of entity size RESCAL is blocks = entity size and
    block size is 1. See the score_emb function for details about the
    implementation.
    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)
        rel_emb_conf_key = self.configuration_key + ".relation_embedder"
        ent_emb_conf_key = rel_emb_conf_key.replace(
            "relation_embedder", "entity_embedder"
        )
        rel_emb_dim = config.get_default(rel_emb_conf_key + ".dim")
        entity_size = config.get_default(ent_emb_conf_key + ".dim")
        if rel_emb_dim > 0:
            raise ValueError(
                "Relation embedding sizes are determined automatically from "
                "sd_rescal.blocks and sd_rescal.block_size or entity_embedding.dim; "
                "do not set manually."
            )

        blocks = config.get_default(self.configuration_key + ".blocks")
        block_size = config.get_default(self.configuration_key + ".block_size")

        if blocks <= 0 and block_size > 0 and entity_size > 0 \
                and entity_size % block_size == 0:
            blocks = entity_size // block_size
        else:
            raise ValueError(
                "If blocks <= 0 then block_size and entity_size have to be "
                "larger than 0 and entity_size has to be dividable by "
                "block_size"
            )

        if block_size <= 0 and blocks > 0 and entity_size > 0 \
                and entity_size % blocks == 0:
            block_size = entity_size // blocks
        else:
            raise ValueError(
                "If block_size <= 0 then blocks and entity_size have to be "
                "larger than 0 and entity_size has to be dividable by "
                "blocks"
            )

        config.set(entity_size, blocks*block_size, log=True)
        config.set(rel_emb_dim, blocks**2*block_size, log=True)

        super().__init__(
            config,
            dataset,
            scorer=SparseDiagonalRescalScorer(config=config, dataset=dataset,
                                              blocks=blocks, block_size=block_size),
            configuration_key=configuration_key
        )
