import torch
from kge import model


class KgeEmbedder(model.KgeBase):
  """
  Base class for all relational model embedders
  """

  def embed_subj(self, subj) -> torch.Tensor:
    """
    Computes the embedding for subject tokens.
    Can be one token for simple lookup or a sequence of tokens.
    """
    raise NotImplementedError

  def embed_rel(self, rel) -> torch.Tensor:
    """
    Computes the embedding for relation tokens.
    Can be one token for simple lookup or a sequence of tokens.
    """
    raise NotImplementedError

  def embed_obj(self, obj) -> torch.Tensor:
    """
    Computes the embedding for object tokens.
    Can be one token for simple lookup or a sequence of tokens.
    """
    raise NotImplementedError

  def get_all_subj(self) -> torch.Tensor:
    """
    Returns all subject embeddings.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def get_all_rel(self) -> torch.Tensor:
    """
    Returns all relation embeddings.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def get_all_obj(self) -> torch.Tensor:
    """
    Returns all object embeddings.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def get_subj(self, subj) -> torch.Tensor:
    """
    Returns one subject embedding.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def get_rel(self, rel) -> torch.Tensor:
    """
    Returns one relation embedding.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def get_obj(self, obj) -> torch.Tensor:
    """
    Returns one object embedding.
    Might trigger a precomputation for embedders which handle sequence of tokens.
    """
    raise NotImplementedError

  def precompute_embeddings_from_tokens(self):
    raise NotImplementedError
