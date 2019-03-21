import torch


class KgeBase(torch.nn.Module):
  """
  Base class for all relational models and embedders
  """

  is_cuda = False
  rel_obj_cache = None
  subj_rel_cache = None

  def cuda(self, device=None):
    super().cuda(device=device)
    self.is_cuda = True

  def cpu(self):
    super().cpu()
    self.is_cuda = False


class KgeModel(KgeBase):
  """
  Base class for all relational models
  """

  def score_spo(self, s, p, o):
    raise NotImplementedError

  def score_sp(self, s, p):
    raise NotImplementedError

  def score_po(self, p, o):
    raise NotImplementedError

  def score_p(self, p):
    raise NotImplementedError
