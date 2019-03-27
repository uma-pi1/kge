import torch
from kge.model.kge_model import KgeModel, KgeEmbedder


class ComplEx(KgeModel):
    """
    ComplEx
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.entity_embedder = KgeEmbedder.create(config, dataset, True)
        self.relation_embedder = KgeEmbedder.create(config, dataset, False)

    def score_spo(self, s, p, o):
        return self._score(s, p, o)

    def score_sp(self, s, p, is_training):
        s = self.entity_embedder.embed(s, is_training)
        p = self.relation_embedder.embed(rel)
        all_objects = self.entity_embedder.embed_all(is_training)
        return self._score(s, p, all_objects, prefix='sp')

    def score_po(self, p, o, is_training):
        all_subjects = self.entity_embedder.embed_all(is_training)
        p = self.relation_embedder.embed(p, is_training)
        o = self.entity_embedder.embed(o, is_training)
        return self._score(all_subjects, p, o, prefix='po')

    def _score(self, s, p, o, prefix=None):
        r"""
        :param s: tensor of size [batch_sz, embedding_size]
        :param p: tensor of size [batch_sz, embedding_size]
        :param o:: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]

        Because the backward is more expensive for n^2 we use the Hadamard form during training

        """
        sub = s.view(-1, s.size(-1))
        rel = p.view(-1, p.size(-1))
        obj = o.view(-1, o.size(-1))

        batch_sz = p.size(0)
        feat_dim = 1

        rel1, rel2 = (t.contiguous() for t in rel.chunk(2, dim=feat_dim))
        obj1, obj2 = (t.contiguous() for t in obj.chunk(2, dim=feat_dim))
        sub_all = torch.cat((sub, sub), dim=feat_dim)
        rel_all = torch.cat((rel1, rel, -rel2,), dim=feat_dim)
        obj_all = torch.cat((obj, obj2, obj1,), dim=feat_dim)

        if prefix:
          if prefix == 'sp':
              out = (sub_all * rel_all).mm(obj_all.transpose(0, 1))
          elif prefix == 'po':
              out = (rel_all * obj_all).mm(sub_all.transpose(0, 1))
          else:
              raise Exception
        else:
            out = (sub_all * obj_all * rel_all).sum(dim=feat_dim)

        return out.view(batch_sz, -1)
