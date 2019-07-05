import torch
import math
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from collections import OrderedDict

class ConvEScorer(RelationalScorer):
    r"""Implementation of the ConvE KGE scorer.

    Must be used with InverseRelationsModel."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)
        self.emb_dim = config.get("lookup_embedder.dim")
        aspect_ratio = config.get("conve.2D_aspect_ratio")
        self.emb_height = math.sqrt(self.emb_dim/aspect_ratio)
        self.emb_width = self.emb_height * aspect_ratio
        if self.emb_dim % self.emb_height or self.emb_dim % self.emb_width:
            raise Exception("Aspect ratio {} does not produce 2D integers for dimension {}.".format(aspect_ratio,
                                                                                                    self.emb_dim))
        self.filter_size = config.get("conve.filter_size")
        self.stride = config.get("conve.stride")
        self.padding = config.get("conve.padding")

        # Comment out from here to test implementation with sequential
        self.feature_map_dropout = torch.nn.Dropout(config.get("conve.feature_map_dropout"))
        self.projection_dropout = torch.nn.Dropout(config.get("conve.projection_dropout"))
        self.convolution = torch.nn.Conv2d(in_channels=1, out_channels=32,
                                           kernel_size=(self.filter_size, self.filter_size),
                                           stride=self.stride, padding=self.padding,
                                           bias=config.get("conve.convolution_bias"))
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        conv_output_height = (((self.emb_height * 2) - self.filter_size + (2 * self.padding))/self.stride) + 1
        conv_output_width = ((self.emb_width - self.filter_size + (2 * self.padding))/self.stride) + 1
        self.projection = torch.nn.Linear(32 * int(conv_output_height * conv_output_width), int(self.emb_dim))
        self.register_parameter('entity_bias', torch.nn.Parameter(torch.zeros(int(self.dataset.num_entities))))
        self.non_linear = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # Comment out until here to test implementation with sequential

        # Implementation with nn.Sequential FROM HERE
        # layers = OrderedDict()
        # layers["convolution"] = torch.nn.Conv2d(in_channels=1, out_channels=32,
        #                                         kernel_size=(self.filter_size, self.filter_size),
        #                                         stride=self.stride, padding=self.padding,
        #                                         bias=config.get("conve.convolution_bias"))
        # layers["batch_norm_1"] = torch.nn.BatchNorm2d(32)
        # layers["non-linear_1"] = torch.nn.ReLU()
        # layers["feature_map_dropout"] = torch.nn.Dropout(config.get("conve.feature_map_dropout"))
        # # TODO add equivalent to out.view(num_triples_to_score, -1) here
        # # apparently not possible, sequential only recommended for simple models
        # conv_output_height = (((self.emb_height * 2) - self.filter_size + (2 * self.padding))/self.stride) + 1
        # conv_output_width = ((self.emb_width - self.filter_size + (2 * self.padding))/self.stride) + 1
        # layers["projection"] = torch.nn.Linear(32 * int(conv_output_height * conv_output_width), int(self.emb_dim))
        # layers["projection_dropout"] = torch.nn.Dropout(config.get("conve.projection_dropout"))
        # layers["batch_norm_2"] = torch.nn.BatchNorm1d(self.emb_dim)
        # layers["non-linear_2"] = torch.nn.ReLU()
        # # TODO add equivalent to applying entity_bias, i.e. tensor.expand_as
        # layers["sigmoid"] = torch.nn.Sigmoid()
        #
        # self.model = torch.nn.Sequential(layers)
        # Implementation with nn.Sequential UNTIL HERE

    # Implementation with nn.Sequential FROM HERE
    # def score_emb_spo(self, s_emb, p_emb, o_emb):
    #     s_emb_2d = s_emb.view(-1, 1, int(self.emb_height), int(self.emb_width))
    #     p_emb_2d = p_emb.view(-1, 1, int(self.emb_height), int(self.emb_width))
    #     stacked_inputs = torch.cat([s_emb_2d, p_emb_2d], 2)
    #     return self.model.forward(stacked_inputs)
    # Implementation with nn.Sequential UNTIL HERE

    # Comment out this function to test implementation with sequential
    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        s_emb_2d = s_emb.view(-1, 1, int(self.emb_height), int(self.emb_width))
        p_emb_2d = p_emb.view(-1, 1, int(self.emb_height), int(self.emb_width))
        stacked_inputs = torch.cat([s_emb_2d, p_emb_2d], 2)
        out = self.convolution(stacked_inputs)
        out = self.bn1(out)
        out = self.non_linear(out)
        out = self.feature_map_dropout(out)
        out = out.view(n, -1)
        out = self.projection(out)
        out = self.projection_dropout(out)
        out = self.bn2(out)
        out = self.non_linear(out)
        if combine == "sp*":
            out = torch.mm(out, o_emb.transpose(1, 0))
        else:
            raise Exception("Combine {} not supported in ConvE's score function".format(combine))
        out += self.entity_bias.expand_as(out)
        out = self.sigmoid(out)

        return out.view(n, -1)


class ConvE(KgeModel):
    r"""Implementation of the ConvE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config,
                         dataset,
                         ConvEScorer(config, dataset),
                         configuration_key=configuration_key)
