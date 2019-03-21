import torch
from kge import embedder


class LookupEmbedder(embedder.KgeEmbedder):

  def __init__(self,
               entity_slot_size,
               relation_slot_size,
               train_data:Union[NegativeSamplingEntityRelationDataset, OneToNEntityRelationDataset],
               entity_embedding_size=None,
               relation_embedding_size=None,
               normalize='',
               dropout=0.0,
               input_dropout=0.0,
               relation_dropout=0.0,
               relation_input_dropout=0.0,
               project_entity=False,
               project_entity_activation='ReLU',
               project_relation=True,
               project_relation_activation=None,
               sparse=False,
               init_std=0.01,
               batch_norm=False,
               l2_reg=0,
               ):
    super().__init__()

    self.train_data=train_data

    if relation_slot_size is None or relation_slot_size <= 0:
      relation_slot_size = entity_slot_size

    self._entity_embedding_size = entity_embedding_size
    if entity_embedding_size is None:
      self._entity_embedding_size = entity_slot_size

    self._relation_embedding_size = relation_embedding_size
    if relation_embedding_size is None:
      self._relation_embedding_size = relation_slot_size

    self.entity_embedding = torch.nn.Embedding(train_data.entities_size, self._entity_embedding_size, sparse=sparse, padding_idx=PAD)
    self.relation_embedding = torch.nn.Embedding(train_data.relations_size, self._relation_embedding_size, sparse=sparse, padding_idx=PAD)

    if hasattr(train_data, "entity_id_sparse_rescaler_map"):
      self.entity_sparse_rescaler_lookup = torch.nn.Embedding(train_data.entities_size, 1, sparse=sparse,)
      self.entity_sparse_rescaler_lookup.weight.data = torch.FloatTensor([v for k,v in sorted(train_data.entity_id_sparse_rescaler_map.items(), key=lambda x:x[0])]).view(-1, 1)
      self.entity_sparse_rescaler_lookup.weight.requires_grad = False
      self.relation_sparse_rescaler_lookup = torch.nn.Embedding(train_data.relations_size, 1, sparse=sparse,)
      self.relation_sparse_rescaler_lookup.weight.data = torch.FloatTensor([v for k,v in sorted(train_data.relation_id_sparse_rescaler_map.items(), key=lambda x:x[0])]).view(-1, 1)
      self.relation_sparse_rescaler_lookup.weight.requires_grad = False

    # Projection for relation / core tensor
    if project_relation:
      project_relation_activation_class = None
      if project_relation_activation:
        project_relation_activation_class = getattr(torch.nn, project_relation_activation)()
      self.relation_projection = Sequential(torch.nn.Linear(self._relation_embedding_size, entity_slot_size ** 2, bias=False), project_relation_activation_class)

    if project_entity:
      project_entity_activation_class = None
      if project_entity_activation:
        project_entity_activation_class = getattr(torch.nn, project_entity_activation)()
      self.subj_projection = Sequential(torch.nn.Linear(entity_slot_size, entity_slot_size, bias=False), project_entity_activation_class)
      self.obj_projection = Sequential(torch.nn.Linear(entity_slot_size, entity_slot_size, bias=False), project_entity_activation_class)

    self.project_entity = project_entity
    self.project_relation = project_relation
    self.slot_size = entity_slot_size
    self.train_data.entities_size = train_data.entities_size
    self.train_data.relations_size = train_data.relations_size
    self.rel_obj_cache = None
    self.subj_rel_cache = None
    self.normalize = normalize

    # Initialize parameters
    torch.nn.init.normal_(self.entity_embedding.weight.data, std=init_std)
    torch.nn.init.normal_(self.relation_embedding.weight.data, std=init_std)
    if project_relation:
      torch.nn.init.xavier_normal_(self.relation_projection.weight.data)

    self.dropout = dropout
    self.input_dropout = input_dropout
    self.relation_dropout = dropout if relation_dropout is None else relation_dropout
    self.relation_input_dropout = input_dropout if relation_input_dropout is None else relation_input_dropout

    self.register_buffer('eye', torch.eye(self.relation_embedding.weight.size(0),self.relation_embedding.weight.size(0)), )

    self.batch_norm = batch_norm
    if self.batch_norm:
      self.bn_e = torch.nn.BatchNorm1d(self._entity_embedding_size)
      self.bn_r = torch.nn.BatchNorm1d(self._relation_embedding_size)

    self.l2_reg = l2_reg
    self._l2_reg_hook = None

  def after_batch_loss_hook(self, epoch):
    if self.training:
      if self.l2_reg > 0:
        result = self._l2_reg_hook
        self._l2_reg_hook = None
        return result
    return None

  def _encode(self, slot_item, embedding, project, input_dropout, dropout, batch_norm=None, lookup=True):
    if lookup:
      slot_item = slot_item.squeeze()
      repr = embedding(slot_item)
    else:
      repr = slot_item
    if input_dropout > 0:
      repr = torch.nn.functional.dropout(repr, p=input_dropout, training=self.training)
    if self.batch_norm:
      repr = batch_norm(repr)
    if project:
      repr = project(repr)
    if self.normalize == 'norm':
      repr = torch.nn.functional.normalize(repr)
    if dropout > 0:
      repr = torch.nn.functional.dropout(repr, p=dropout, training=self.training)
    if self.training and self.l2_reg > 0:
      _l2_reg_hook = repr
      if self.dropout > 0:
        _l2_reg_hook  = _l2_reg_hook / self.dropout
      _l2_reg_hook = self.l2_reg*_l2_reg_hook.abs().pow(3).sum()
      if self._l2_reg_hook is None:
        self._l2_reg_hook = _l2_reg_hook
      else:
        self._l2_reg_hook = self._l2_reg_hook + _l2_reg_hook
    return repr

  def embed_rel(self, rel, lookup=True):
    return self._encode(rel,
                        self.relation_embedding,
                        self.relation_projection if self.project_relation else None,
                        self.relation_input_dropout,
                        self.relation_dropout,
                        self.bn_r if self.batch_norm else None,
                        lookup=lookup
                        )

  def embed_subj(self, subj, lookup=True):
    return self._encode(subj,
                        self.entity_embedding,
                        self.subj_projection if self.project_entity else None,
                        self.input_dropout,
                        self.dropout,
                        self.bn_e if self.batch_norm else None,
                        lookup=lookup
                        )

  def embed_obj(self, obj, lookup=True):
    return self._encode(obj,
                        self.entity_embedding,
                        self.obj_projection if self.project_entity else None,
                        self.input_dropout,
                        self.dropout,
                        self.bn_e if self.batch_norm else None,
                        lookup=lookup
                        )

  def _get_all_(self, min_size, encode_func, embedding, as_variable=False, include_special_items=False):
    if include_special_items:
      min_offset = 0
    else:
      min_offset = min_size
    result = encode_func(embedding.weight[min_offset:].contiguous(), lookup=False)
    if not as_variable:
      result = result.data
    return result

  def get_all_rel(self, as_variable=False, include_special_items=False):
    return self._get_all_(self.train_data.min_relations_size, self.encode_rel, self.relation_embedding,as_variable, include_special_items)

  def get_all_subj(self, as_variable=False, include_special_items=False):
    return self._get_all_(self.train_data.min_entities_size, self.encode_subj, self.entity_embedding,as_variable, include_special_items)

  def get_all_obj(self, as_variable=False, include_special_items=False):
    return self._get_all_(self.train_data.min_entities_size, self.encode_obj, self.entity_embedding,as_variable, include_special_items)

  def _get_(self, encode_func, id, as_variable=False):
    id = torch.LongTensor([id]).pin_memory()
    if self.is_cuda:
      id = id.cuda()
    result = encode_func(id)
    if not as_variable:
      result = result.data
    return result

  def get_subj(self, subj, as_variable=False):
    return self._get_(self.encode_subj, subj, as_variable)

  def get_rel(self, rel, as_variable=False):
    return self._get_(self.encode_rel, rel, as_variable)

  def get_obj(self, obj, as_variable=False):
    return self._get_(self.encode_obj, obj, as_variable)

  def get_slot_size(self):
    return self.slot_size
