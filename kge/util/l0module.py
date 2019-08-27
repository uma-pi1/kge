import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class L0(nn.Module):
    pass


class _L0Norm(L0):
    def __init__(
        self,
        origin,
        loc_mean=0,
        loc_sdev=1e-2,
        beta=2 / 3,
        gamma=-0.1,
        zeta=1.1,
        fix_temp=True,
    ):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.loc_mean = loc_mean
        self.loc_sdev = loc_sdev
        self.gamma = gamma
        self.beta = beta
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "loc_mean="
            + str(self.loc_mean)
            + ", loc_sdev="
            + str(self.loc_sdev)
            + ", beta="
            + str(self.beta)
            + ", gamma="
            + str(self.gamma)
            + ", zeta="
            + str(self.zeta)
            + str("\n\t")
            + self._origin.__repr__()
            + "\n)"
        )

    def _get_mask(self):
        min_eps = 1e-8
        max_eps = 10
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            self.loc.data = self.loc.data.clamp(min=min_eps, max=max_eps)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + torch.log(self.loc)) / self.temp
            )
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(
                torch.log(self.loc) - self.temp * self.gamma_zeta_ratio
            ).sum()
            if isnan(penalty.data):
                print(self.loc)
        else:
            loc = Variable(self.loc.data.clamp(min=min_eps, max=max_eps))
            s = torch.sigmoid(torch.log(loc)) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


# no clamping; TODO document
class _L0Norm_orig(L0):
    def __init__(
        self,
        origin,
        loc_mean=0,
        loc_sdev=1e-2,
        beta=2 / 3,
        gamma=-0.1,
        zeta=1.1,
        fix_temp=True,
    ):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm_orig, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.loc_mean = loc_mean
        self.loc_sdev = loc_sdev
        self.gamma = gamma
        self.beta = beta
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "loc_mean="
            + str(self.loc_mean)
            + ", loc_sdev="
            + str(self.loc_sdev)
            + ", beta="
            + str(self.beta)
            + ", gamma="
            + str(self.gamma)
            + ", zeta="
            + str(self.zeta)
            + str("\n\t")
            + self._origin.__repr__()
            + "\n)"
        )

    def _get_mask(self):

        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty


def isnan(x):
    return (x != x).long().sum() > 0


class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(
            nn.Linear(in_features, out_features, bias=bias), **kwargs
        )

    def forward(self, input):
        mask, penalty = self._get_mask()
        return (
            F.linear(input, self._origin.weight * mask, self._origin.bias),
            penalty,
            ((mask > 0).float().sum() / mask.numel()).item(),
        )


class L0Linear_orig(_L0Norm_orig):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear_orig, self).__init__(
            nn.Linear(in_features, out_features, bias=bias), **kwargs
        )

    def forward(self, input):
        mask, penalty = self._get_mask()
        return (
            F.linear(input, self._origin.weight * mask, self._origin.bias),
            penalty,
            ((mask > 0).float().sum() / mask.numel()).item(),
        )
