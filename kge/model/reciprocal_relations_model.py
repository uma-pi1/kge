import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class ReciprocalRelationsModel(KgeModel):
    """Modifies a base model to use different relation embeddings for predicting subject and object.

    This implements the reciprocal relations training procedure of [TODO cite ConvE].
    Note that this model cannot be used to score a single triple, but only to rank sp_
    or _po questions.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        # Using a dataset with twice the number of relations to initialize base model
        alt_dataset = dataset.shallow_copy()
        alt_dataset._num_relations = dataset.num_relations() * 2
        base_model = KgeModel.create(
            config=config,
            dataset=alt_dataset,
            configuration_key=self.configuration_key + ".base_model",
            init_for_load_only=init_for_load_only,
        )

        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=base_model.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only,
        )
        self._base_model = base_model
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        if direction == "o":
            return super().score_spo(s, p, o, "o")
        elif direction == "s":
            return super().score_spo(o, p + self.dataset.num_relations(), s, "o")
        else:
            raise Exception(
                "The reciprocal relations model cannot compute "
                "undirected spo scores."
            )

    def score_po(self, p, o, s=None):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(o, p, s, combine="sp_")

    def score_so(self, s, o, p=None):
        raise Exception("The reciprocal relations model cannot score relations.")

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ) -> torch.Tensor:
        s = self.get_s_embedder().embed(s)
        p_inv = self.get_p_embedder().embed(p + self.dataset.num_relations())
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
            po_scores = self._scorer.score_emb(o, p_inv, all_entities, combine="sp_")
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp_")
            po_scores = self._scorer.score_emb(o, p_inv, all_subjects, combine="sp_")
        return torch.cat((sp_scores, po_scores), dim=1)
