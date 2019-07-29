import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class InverseRelationsModel(KgeModel):
    """Modifies a base model to use different relation embeddings for predicting subject and object.

    This implements the inverse relations training procedure of [TODO cite ConvE]. Note that this model
    cannot be used to score a single triple, but only to rank sp* or *po questions.

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        # Using a dataset with twice the number of relations to initialize base model
        alt_dataset = Dataset(dataset.config,
                              dataset.num_entities,
                              dataset.entities,
                              dataset.num_relations * 2,
                              dataset.relations,
                              dataset.train,
                              dataset.train_meta,
                              dataset.valid,
                              dataset.valid_meta,
                              dataset.test,
                              dataset.test_meta,
                              )
        base_model = KgeModel.create(config,
                                     alt_dataset,
                                     self.configuration_key + ".base_model")

        # Initialize this model
        super().__init__(config, dataset, base_model.get_scorer(), initialize_embedders=False)
        self._base_model = base_model
        # TODO change entity_embedder assignment to sub and obj embedders when support for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs)

    # def score_spo(self, s, p, o):
    #     raise Exception("The inverse relations model cannot compute spo scores.")

    def score_po(self, p, o, s=None):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p + self.dataset.num_relations)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, s, combine="sp*")

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
