import torch
from kge import model


class ComplexRelationScorer(model.KgeModel):

  def triple_score(self, subj, rel, obj, drop_relation=False):
    return self._score(subj, rel, obj,)

  def _score(self, subj, rel, obj, prefix=False, drop_relation=False, sp=None, po=None):
    r"""
    :param subj: tensor of size [batch_sz, embedding_size]
    :param rel: tensor of size [batch_sz, embedding_size]
    :param obj: tensor of size [batch_sz, embedding_size]
    :return: score tensor of size [batch_sz, 1]

    Because the backward is more expensive for n^2 we use the Hadamard form during training

    """
    # if not self.training:
    #     print(subj.size(), rel.size(), obj.size())
    batch_sz = rel.size(0)

    subj = subj.view(-1, subj.size(-1))
    rel = rel.view(-1, rel.size(-1))
    obj = obj.view(-1, obj.size(-1))

    feat_dim = 1

    rel1, rel2 = (t.contiguous() for t in rel.chunk(2, dim=feat_dim))
    obj1, obj2 = (t.contiguous() for t in obj.chunk(2, dim=feat_dim))
    subj_all = torch.cat((subj, subj), dim=feat_dim)
    rel_all = torch.cat((rel1, rel, -rel2,), dim=feat_dim)
    obj_all = torch.cat((obj, obj2, obj1,), dim=feat_dim)

    if prefix:
      if sp:
        out = (subj_all * rel_all).mm(obj_all.transpose(0,1))
      elif po:
        out = (rel_all * obj_all).mm(subj_all.transpose(0,1))
      else:
        raise Exception
    else:
      out = (subj_all * obj_all * rel_all).sum(dim=feat_dim)

    return out.view(batch_sz, -1)

  def sp_prefix_score(self, subj=None, rel=None, many_obj=None):
    subj = self.encode_subj(subj)
    rel = self.encode_rel(rel)
    if many_obj is None:
      many_obj = self.get_all_obj(as_variable=self.training)
    return self._score(subj, rel, many_obj, prefix=True, sp=True, po=False)

  def po_prefix_score(self, rel=None, obj=None, many_subj=None):
    # print(rel, obj)
    if many_subj is None:
      many_subj = self.get_all_subj(as_variable=self.training)
    rel = self.encode_rel(rel)
    obj = self.encode_obj(obj)
    return self._score(many_subj, rel, obj, prefix=True, sp=False, po=True)

  def precompute_batch_shared_inputs(self, entity_ids):
    return (self.encode_obj(entity_ids),)
