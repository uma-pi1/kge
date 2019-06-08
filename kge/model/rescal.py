import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel, KgeEmbedder, KgeBase


class RESCALScorer(RelationalScorer):
    r"""Implementation of the RESCAL KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb(self, s_emb: torch.Tensor, p_emb: torch.Tensor, o_emb: torch.Tensor, combine: str):
        batch_size = p_emb.size(0)
        entity_size = s_emb.size(-1)

        p_emb = p_emb.view(-1, entity_size, entity_size)

        if combine == "spo":
            out = (s_emb.unsqueeze(1).bmm(p_emb).view(batch_size, entity_size) * o_emb).sum(dim=-1)
        elif combine == "sp*":
            out = (s_emb.view(-1, 1, entity_size).bmm(p_emb)).view(-1, entity_size).mm(o_emb.transpose(0, 1))
        elif combine == "*po":
            out = p_emb.bmm(o_emb.view(-1, entity_size, 1)).view(-1, entity_size).mm(s_emb.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(batch_size, -1)


class RESCAL(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(
            config,
            dataset,
            scorer=RESCALScorer(config=config, dataset=dataset))

        config.set(
            config.get("model") + ".relation_embedder.dim",
            config.get(config.get("model") + ".entity_embedder.dim")**2
        )

        #: Embedder used for relations
        self._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            config.get("model") + ".relation_embedder",
            dataset.num_relations,
        )
