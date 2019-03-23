import torch
from kge.model import ComplEx


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
		super().__init__(config, dataset)

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
		if config.get('model.type') == 'complex':
			return ComplEx(config, dataset)
		else:
			# perhaps TODO: try class with specified name -> extensibility
			raise ValueError('model.type')

			# TODO I/O


class KgeEmbedder(KgeBase):
	"""
	Base class for all relational model embedders
	"""

	def __init__(self, config, dataset):
		super().__init__(config, dataset)

	def create(config, dataset, for_entities):
		"""Factory method for embedder creation."""
		# entity embedder
		if config.get('model.entity_embedder') == 'lookup':
			entity_embedder = LookupEmbedder(config, dataset)
		else:
			# perhaps TODO: try class with specified name -> extensibility
			raise ValueError('model.entity_embedder')

		# relation embedder
		if config.get('model.relation_embedder') == 'lookup':
			relation_embedder = LookupEmbedder(config, dataset)
		else:
			# perhaps TODO: try class with specified name -> extensibility
			raise ValueError('model.relation_embedder')

		return entity_embedder, relation_embedder

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
