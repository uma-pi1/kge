import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, KgeEmbedder, RelationalScorer


class InverseModel(KgeModel):

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset, None, initialize_embedders=False)

        # Set base_model as base_model in config
        config.set("model", config.get("inverse_model.base_model"))

        # Initialize base model
        self._base_model = KgeModel.create(config, dataset)

        # Change relation embedder in base model
        # TODO: if I had access to the initialize_embedders flag, I'd use it here
        # Advantage would be not creating a relation embedder twice
        # But adding such access means changing the signature of the constructors of all existing models
        self._base_model._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            config.get("model") + ".relation_embedder",
            dataset.num_relations * 2,
            )

        # Initialize this model with the scorer of base_model
        self._scorer = self._base_model.get_scorer()

    def prepare_job(self, job, **kwargs):
        # TODO how to handle this when parent class has no embedders?
        # super().prepare_job(job, **kwargs)
        self._base_model._entity_embedder.prepare_job(job, **kwargs)
        self._base_model._relation_embedder.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return (
            # TODO how to handle this when parent class has no embedders?
            # super().penalty(**kwargs) +
            self._base_model._entity_embedder.penalty(**kwargs) +
            self._base_model._relation_embedder.penalty(**kwargs)
        )

    def get_s_embedder(self) -> KgeEmbedder:
        return self._base_model._entity_embedder

    def get_o_embedder(self) -> KgeEmbedder:
        return self._base_model._entity_embedder

    def get_p_embedder(self) -> KgeEmbedder:
        return self._base_model._relation_embedder

    def get_scorer(self) -> RelationalScorer:
        return self._scorer

    def score_spo(self, s, p, o):
        raise Exception("The inverse model cannot compute spo scores.")

    def score_po(self, p, o):
        all_subjects = self.get_s_embedder().embed_all()
        p = self.get_p_embedder().embed(p + self.dataset.num_relations)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, all_subjects, combine="sp*")

    def score_sp_po(self, s, p, o):
        s = self.get_s_embedder().embed(s)
        p_inv = self.get_p_embedder().embed(p + self.dataset.num_relations)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp*")
            po_scores = self._scorer.score_emb(o, p_inv, all_entities, combine="sp*")
        else:
            all_objects = self.get_o_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp*")
            all_subjects = self.get_s_embedder().embed_all()
            po_scores = self._scorer.score_emb(o, p_inv, all_subjects, combine="sp*")
        return torch.cat((sp_scores, po_scores), dim=1)
