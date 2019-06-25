import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


# TODO: change implementation to suit framework
# For now everything is here
# Required decisions:
#   Where to add batch normalization before input dropout?
#       Cannot do this by overriding score_sp and others of KgeModel,because ConvE uses the inverse_model
#       Inverse model already has its own version of score_sp and others
#   Where to do the conversion of embeddings to 2D?
#       Again, cannot do this by overriding score_sp of KgeModel,because ConvE uses the inverse_model
#   What is the purpose of the parameter b that is manually added to the model?
#   We should implement Xavier initialization, as original model uses that
#   We should implement label smoothing, as original model they uses that

class ConvEScorer(RelationalScorer):
    r"""Implementation of the ConvE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)
        # TODO how to do this not hardcoded?
        self.emb_dim = config.get("lookup_embedder.dim")
        self.emb_height = config.get("conve.embedding_height")
        self.emb_width = self.emb_dim / self.emb_height
        self.filter_size = config.get("conve.filter_size")
        self.stride = config.get("conve.stride")
        self.padding = config.get("conve.padding")
        # TODO for now input dropout is here until we add batch normalization to lookup embedder
        self.input_dropout = torch.nn.Dropout(config.get("conve.input_dropout"))
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
        # TODO not sure why they use this manually added parameter b
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(int(self.dataset.num_entities))))
        self.non_linear = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        s_emb_2d = s_emb.view(-1, 1, self.emb_height, int(self.emb_width))
        p_emb_2d = p_emb.view(-1, 1, self.emb_height, int(self.emb_width))
        stacked_inputs = torch.cat([s_emb_2d, p_emb_2d], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        out = self.input_dropout(stacked_inputs)
        out = self.convolution(out)
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
        elif combine == "spo":
            out = (out * o_emb).sum(dim=1)
        else:
            raise Exception("Combine {} not supported in ConvE's score function".format(combine))
        out += self.b.expand_as(out)
        out = self.sigmoid(out)

        return out.view(n, -1)


class ConvE(KgeModel):
    r"""Implementation of the ConvE KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset, ConvEScorer(config, dataset))
