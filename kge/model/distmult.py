from kge.model.kge_model import KgeModel, KgeEmbedder


class DistMult(KgeModel):
    """
    DistMult
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def _score(self, s, p, o, prefix=None):
        r"""
        :param s: tensor of size [batch_size, embedding_size]
        :param p: tensor of size [batch_size, embedding_size]
        :param o:: tensor of size [batch_size, embedding_size]
        :return: score tensor of size [batch_size, 1]
        """
        sub = s.view(-1, s.size(-1))
        rel = p.view(-1, p.size(-1))
        obj = o.view(-1, o.size(-1))

        batch_size = p.size(0)
        feat_dim = 1

        if prefix:
            if prefix == 'sp':
                out = (sub * rel).mm(obj.transpose(0, 1))
            elif prefix == 'po':
                out = (rel * obj).mm(sub.transpose(0, 1))
            else:
                raise Exception
        else:
            out = (sub * obj * rel).sum(dim=feat_dim)

        return out.view(batch_size, -1)
