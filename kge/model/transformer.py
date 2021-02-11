import torch
import torch.nn
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class TransformerScorer(RelationalScorer):
    r"""Scorer that uses a plain Transformer encoder.

    Concatenates (1) CLS embedding, (2) subject entity embedding (one per entity) +
    subject type embedding, (3) relation embedding (one per relation) + relation type
    embedding. Then runs transformer encoder and takes dot product with transformed CLS
    emebdding and object entity embedding to produce score.

    Must be used with ReciprocalRelationsModel.

    Based on the "No context" model from:

    HittER: Hierarchical Transformers for Knowledge Graph Embeddings
    Sanxing Chen, Xiaodong Liu, Jianfeng Gao, Jian Jiao, Ruofei Zhang and Yangfeng Ji
    https://arxiv.org/abs/2008.12813

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.emb_dim = self.get_option("entity_embedder.dim")

        # the CLS embedding
        self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.cls_emb)
        # the type embeddings
        self.sub_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.sub_type_emb)
        self.rel_type_emb = torch.nn.parameter.Parameter(torch.zeros(self.emb_dim))
        self.initialize(self.rel_type_emb)

        dropout = self.get_option("encoder.dropout")
        if dropout < 0.0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.encoder.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0.0

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.get_option("encoder.nhead"),
            dim_feedforward=self.get_option("encoder.dim_feedforward"),
            dropout=dropout,
            activation=self.get_option("encoder.activation"),
        )
        self.encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.get_option("encoder.num_layers")
        )
        for layer in self.encoder.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)
            self.initialize(layer.self_attn.out_proj.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise ValueError(
                "Combine {} not supported in Transformer's score function".format(
                    combine
                )
            )

        # transform the sp pairs
        batch_size = len(s_emb)
        out = self.encoder.forward(
            torch.stack(
                (
                    self.cls_emb.repeat((batch_size, 1)),
                    s_emb + self.sub_type_emb.unsqueeze(0),
                    p_emb + self.rel_type_emb.unsqueeze(0),
                ),
                dim=0,
            )
        )  # SxNxE = 3 x batch_size x emb_size

        # pick the transformed CLS embeddings
        out = out[0, ::]

        # now take dot product
        if combine == "sp_":
            out = torch.mm(out, o_emb.transpose(1, 0))
        elif combine == "spo":
            out = (out * o_emb).sum(-1)
        else:
            raise Exception("can't happen")

        # all done
        return out.view(batch_size, -1)


class Transformer(KgeModel):
    r"""Implementation of the Transformer KGE model."""

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
            scorer=TransformerScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If Transformer is wrapped in a reciprocal relations model, this will always be
        # the case.
        if direction == "o":
            super().score_spo(s, p, o, direction)
        else:
            raise ValueError("Transformer can only score objects")
