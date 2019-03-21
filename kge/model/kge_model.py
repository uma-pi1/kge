import torch

class KgeBase(torch.nn.Module):
  """
  Base class for all relational models and embedders
  """
  is_cuda = False

  def __init__(self, config, dataset):
    self.config = config
    self.dataset = dataset

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
  def __init__(self, config, dataset):
    super(KgeModel,self).__init__(config, dataset)

  def score_spo(self, s, p, o):
    raise NotImplementedError

  def score_sp(self, s, p):
    raise NotImplementedError

  def score_po(self, p, o):
    raise NotImplementedError

  def score_p(self, p):
    raise NotImplementedError

  def create(config, dataset):
    """Factory method for model creation."""
    # TODO return somethign useful ;)
    # embedder creation here as well
    return KgeModel(config, dataset)
    pass

  # TODO I/O

class KgeEmbedder(KgeBase):
  """
  Base class for all relational model embedders
  """

  def __init__(self, config, dataset):
    super(KgeModel,self).__init__(config, dataset)

  def create(config, dataset, for_entities):
    """Factory method for embedder creation."""
    # TODO return somethign useful ;)
    return KgeEmbedder(config, dataset)
    pass

  def embed(self, i) -> torch.Tensor:
    """
    Computes the embedding.
    """
    raise NotImplementedError

  def get_all(self) -> torch.Tensor:
    """
    Returns all embeddings.
    """
    raise NotImplementedError

  # TODO I/O
