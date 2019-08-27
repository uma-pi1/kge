import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F

class TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        # TODO drop [None,:,:] everywhere once the cdist bug (
        # https://github.com/pytorch/pytorch/issues/22353) has been fixed. The fix is
        # https://github.com/pytorch/pytorch/commit/c33adf539c82fe26ed678a2aea4427fbbcbd7c97,
        # but has not been released (released in pytorch 1.2.1?).
        n = p_emb.size(0)
        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp*":
            sp_emb = s_emb + p_emb
            out = -torch.cdist(sp_emb[None,:,:], o_emb[None,:,:], p=self._norm)
        elif combine == "*po":
            po_emb = o_emb - p_emb
            out = -torch.cdist(po_emb[None,:,:], s_emb[None,:,:], p=self._norm)
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')
        return out.view(n, -1)


# # TranseE using TorchScript for PT 1.2
#
#
# class TransEScorer(RelationalScorer, torch.jit.ScriptModule):
#     r"""Implementation of the TransE KGE scorer."""
#
#     def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
#         super().__init__(config, dataset, configuration_key=configuration_key)
#         self.combines = {"spo": 0, "sp*": 1, "*po": 2}
#
#     def score_emb(self, s_emb, p_emb, o_emb, combine: int):
#         return self._score_emb(s_emb, p_emb, o_emb, self.combines[combine], self.config.get("job.device"), self.config.get("transe.l_norm"))
#
#     @torch.jit.script_method
#     def _score_emb(self, s_emb, p_emb, o_emb, combine:int, device:str, norm: float):
#         n = p_emb.size(0)
#         if combine == 0:
#             out = -torch.norm(s_emb + p_emb - o_emb, norm, 1)
#         elif combine == 1:
#             out = torch.zeros(n, o_emb.size(0)).to(device)
#             for i in range(n):
#                 out[i, :] = -torch.norm(
#                     (s_emb[i, :] + p_emb[i, :]) - o_emb, norm, 1
#                 )
#         elif combine == 2:
#             out = torch.zeros(n, s_emb.size(0)).to(device)
#             for i in range(n):
#                 out[i, :] = -torch.norm(
#                     s_emb + (p_emb[i, :] - o_emb[i, :]), norm, 1
#                 )
#         else:
#             out = torch.zeros(1,1)
#             raise ValueError('cannot handle combine="{}".format(combine)')
#
#         return out.view(n, -1)


class TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        super().__init__(
            config,
            dataset,
            TransEScorer(config, dataset, self.configuration_key),
            configuration_key=configuration_key,
        )
