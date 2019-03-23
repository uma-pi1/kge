import torch
from kge.model import KgeEmbedder


class LookupEmbedder(KgeEmbedder):
	def __init__(self, config, dataset, embed_entities=True):
		super().__init__(config, dataset)

		if config.get('job.type') == 'train':
			self.training = True
		else:
			self.training = False
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.input_dropout = input_dropout
		self.l2_reg = l2_reg
		self._l2_reg_hook = None

		# TODO: what is this?
		self.register_buffer('eye',
												 torch.eye(self.relation_embedding.weight.size(0), self.relation_embedding.weight.size(0)), )

		if embed_entities:
			self.embeddings = torch.nn.Embedding(dataset.num_entities, config.get('model.dim'),
																					 sparse=config.get('model.sparse'))
			self.normalize = config.get('model.normalize_entities')
			if self.batch_norm:
				self.bn = torch.nn.BatchNorm1d(dataset.num_entities)
		else:
			self.embeddings = torch.nn.Embedding(dataset.num_relations, config.get('model.dim'),
																					 sparse=config.get('model.sparse'))
			self.normalize = config.get('model.normalize_relations')
			if self.batch_norm:
				self.bn = torch.nn.BatchNorm1d(dataset.num_relations)

		# Initialize parameters
		torch.nn.init.normal_(self.embeddings.weight.data, std=config.get('model.init_std'))

		def after_batch_loss_hook(self, epoch):
			if self.training:
				if self.l2_reg > 0:
					result = self._l2_reg_hook
					self._l2_reg_hook = None
					return result
			return None

	def _encode(self, embeddings, input_dropout, dropout, batch_norm=None):
		if input_dropout > 0:
			embeddings = torch.nn.functional.dropout(embeddings, p=input_dropout, training=self.training)
		if self.batch_norm:
			embeddings = batch_norm(embeddings)
		if self.normalize == 'norm':
			embeddings = torch.nn.functional.normalize(embeddings)
		if dropout > 0:
			embeddings = torch.nn.functional.dropout(embeddings, p=dropout, training=self.training)
		if self.training and self.l2_reg > 0:
			_l2_reg_hook = embeddings
			if self.dropout > 0:
				_l2_reg_hook = _l2_reg_hook / self.dropout
			_l2_reg_hook = self.l2_reg * _l2_reg_hook.abs().pow(3).sum()
			if self._l2_reg_hook is None:
				self._l2_reg_hook = _l2_reg_hook
			else:
				self._l2_reg_hook = self._l2_reg_hook + _l2_reg_hook
		return embeddings

	def embed(self, embeddings):
		return self._encode(embeddings,
												self.input_dropout,
												self.dropout,
												self.bn if self.batch_norm else None)

	def _get_all_(self, encode_func, embedding, as_variable=False):
		result = encode_func(embedding.weight[min_offset:].contiguous(), lookup=False)
		if not as_variable:
			result = result.data
		return result

	def get_all(self, as_variable=False):
		return self._get_all_(self.embed, self.embeddings, as_variable)
