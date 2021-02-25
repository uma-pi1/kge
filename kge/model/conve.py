import torch
from torch import Tensor
import math

from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class ConvEScorer(RelationalScorer):
    r"""Implementation of the ConvE KGE scorer.

    Must be used with ReciprocalRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        # self.configuration_key = configuration_key
        self.emb_dim = self.get_option("entity_embedder.dim") - 1
        aspect_ratio = self.get_option("2D_aspect_ratio")
        self.emb_height = math.sqrt(self.emb_dim / aspect_ratio)
        self.emb_width = self.emb_height * aspect_ratio

        # round embedding dimension to match aspect ratio
        rounded_height = math.ceil(self.emb_height)
        if self.get_option("round_dim") and rounded_height != self.emb_height:
            self.emb_height = rounded_height
            self.emb_width = self.emb_height * aspect_ratio
            self.emb_dim = self.emb_height * self.emb_width
            self.set_option("entity_embedder.dim", self.emb_dim + 1, log=True)
            self.set_option("relation_embedder.dim", self.emb_dim + 1, log=True)
            config.log(
                "Rounded embedding dimension up to {} to match given aspect ratio.".format(
                    self.emb_dim
                )
            )
        elif self.emb_dim % self.emb_height or self.emb_dim % self.emb_width:
            raise Exception(
                (
                    "Embedding dimension {} incompatible with aspect ratio {}; "
                    "width ({}) or height ({}) is not integer. "
                    "Adapt dimension or set conve.round_dim=true"
                ).format(self.emb_dim, aspect_ratio, self.emb_width, self.emb_height)
            )

        self.filter_size = self.get_option("filter_size")
        self.stride = self.get_option("stride")
        self.padding = self.get_option("padding")
        self.feature_map_dropout = torch.nn.Dropout2d(
            self.get_option("feature_map_dropout")
        )
        self.projection_dropout = torch.nn.Dropout(
            self.get_option("projection_dropout")
        )
        self.convolution = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(self.filter_size, self.filter_size),
            stride=self.stride,
            padding=self.padding,
            bias=self.get_option("convolution_bias"),
        )
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim, affine=False)
        conv_output_height = (
            ((self.emb_height * 2) - self.filter_size + (2 * self.padding))
            / self.stride
        ) + 1
        conv_output_width = (
            (self.emb_width - self.filter_size + (2 * self.padding)) / self.stride
        ) + 1
        self.projection = torch.nn.Linear(
            32 * int(conv_output_height * conv_output_width), int(self.emb_dim)
        )
        self.non_linear = torch.nn.ReLU()

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if combine not in ["sp_", "spo"]:
            raise Exception(
                "Combine {} not supported in ConvE's score function".format(combine)
            )

        batch_size = p_emb.size(0)
        s_emb_2d = s_emb[:, 1:].view(-1, 1, int(self.emb_height), int(self.emb_width))
        p_emb_2d = p_emb[:, 1:].view(-1, 1, int(self.emb_height), int(self.emb_width))
        stacked_inputs = torch.cat([s_emb_2d, p_emb_2d], 2)
        out = self.convolution(stacked_inputs)
        out = self.bn1(out)
        out = self.non_linear(out)
        out = self.feature_map_dropout(out)
        out = out.view(batch_size, -1)
        out = self.projection(out)
        out = self.projection_dropout(out)
        out = self.bn2(out)
        out = self.non_linear(out)
        if combine == "sp_":
            out = torch.mm(out, o_emb[:, 1:].transpose(1, 0))
        else:
            assert combine == "spo"
            out = (out * o_emb[:, 1:]).sum(-1)
        out += o_emb[:, 0]

        return out.view(batch_size, -1)


class ConvE(KgeModel):
    r"""Implementation of the ConvE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        # HACK to add bias terms to embeddings
        self.set_option(
            "entity_embedder.dim", self.get_option("entity_embedder.dim") + 1
        )
        self.set_option(
            "relation_embedder.dim", self.get_option("relation_embedder.dim") + 1
        )
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=ConvEScorer(config, dataset, self.configuration_key),
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        # UNDO hack
        self.set_option(
            "entity_embedder.dim", self.get_option("entity_embedder.dim") - 1
        )
        self.set_option(
            "relation_embedder.dim", self.get_option("relation_embedder.dim") - 1
        )

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # We overwrite this method to ensure that ConvE only predicts towards objects.
        # If ConvE is wrapped in a reciprocal relations model, this will always be the
        # case.
        if direction == "o":
            return super().score_spo(s, p, o, direction)
        else:
            raise ValueError("ConvE can only score objects")
