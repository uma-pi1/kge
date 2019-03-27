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


    def _score(self, s, p, o, prefix=None, is_training=False):
        r"""
        :param s: tensor of size [batch_size, embedding_size]
        :param p: tensor of size [batch_size, embedding_size]
        :param o:: tensor of size [batch_size, embedding_size]
        :return: score tensor of size [batch_size, 1]

        Because the backward is more expensive for n^2 we use the Hadamard form during training

        """
        sub = s.view(-1, s.size(-1))
        rel = p.view(-1, p.size(-1))
        obj = o.view(-1, o.size(-1))

        batch_size = p.size(0)
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

        return out.view(batch_size, -1)
