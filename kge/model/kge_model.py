from kge import Config
import copy
import torch.nn
import importlib


class KgeBase(torch.nn.Module):
    """
    Base class for all relational models and embedders.
    """

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

    def initialize(self, what, initialize, initialize_arg):
        if initialize == "normal":
            torch.nn.init.normal_(what, std=initialize_arg)
        else:
            raise ValueError("initialize")


class KgeModel(KgeBase):
    """Base class for all KGE models.

    KGE models take as input the embeddings of (subject, predicate, object)-triple and
    produce a score.

    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    @staticmethod
    def create(config, dataset):
        """Factory method for model creation."""

        model = None
        try:
            model_name = config.get("model")
            class_name = config.get(model_name + ".class_name")
            module = importlib.import_module("kge.model")
            model = getattr(module, class_name)(config, dataset)
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for model {}".format(
                    class_name, model_name
                )
            )

        model.to(config.get("job.device"))
        return model

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        """Scores a set of triples specified by their embeddings.

        `s_emb`, `p_emb`, and `o_emb` are tensors of size :math:`n_s\\times d_e`,
        :math:`n_p\times d_r`, and :math:`n_o\times d_e`, where :math:`d_e` and
        :math:`d_r` are the sizes of the entity and relation embeddings, respectively.

        The provided embeddings are combined based on the value of `combine`. Common
        values are ``"spo"``, ``"sp*"``, and ``"*po"``. Not all models may support all
        combinations.

        When `combine` is ``"spo"``, then embeddings are combined row-wise. In this
        case, it is required that :math:`n_s=n_p=n_o=n`. The output is a :math`n\\times
        1` tensor, in which the :math:`i`-th entry holds the score of the embedding
        triple $(s_i, p_i, o_i)$.

        When `combine` is ``"sp*"``, the subjects and predicates are taken row-wise and
        subsequently combined with all objects. In this case, it is required that
        :math:`n_s=n_p=n`. The output is a :math`n\\times n_o` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple $(s_i, p_i,
        o_j)$.

        When `combine` is `"*po"`, predicates and objects are taken row-wise and
        subsequently combined with all subjects. In this case, it is required that
        :math:`n_p=n_o=n`. The output is a :math`n\\times n_s` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple $(s_j, p_i,
        o_i)$.

        """
        raise NotImplementedError


class ClosedKgeModel(KgeModel):
    """Base class of KGE models that embed a fixed set of entities and relations.

    Each entity and each relation is associated with an index. This class uses
    `KgeEmbedder` to associate each subject, relation, and object index with an
    embedding.

    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # TODO add support for three embedders
        self._entity_embedder = KgeEmbedder.create(
            config,
            dataset,
            config.get("model") + ".entity_embedder",
            dataset.num_entities,
        )
        self._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            config.get("model") + ".relation_embedder",
            dataset.num_relations,
        )

    def get_s_embedder(self):
        return self._entity_embedder

    def get_o_embedder(self):
        return self._entity_embedder

    def get_p_embedder(self):
        return self._relation_embedder

    def score_spo(self, s, p, o):
        """Compute scores for a set of triples.

        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)``.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self.score_emb(s, p, o, combine="spo")

    def score_sp(self, s, p):
        """Compute scores for triples formed from a set of sp-pairs and all objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        all_objects = self.get_o_embedder().embed_all()
        return self.score_emb(s, p, all_objects, combine="sp*")

    def score_po(self, p, o):
        """Compute scores for triples formed from a set of po-pairs and all subjects.

        `p` and `o` are vectors of common size :math:`n`, holding the indexes of the
        relations and objects to score.

        Returns an :math:`n\\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(j,
        p_i, o_i)`.

        """
        all_subjects = self.get_s_embedder().embed_all()
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self.score_emb(all_subjects, p, o, combine="*po")

    def score_sp_po(self, s, p, o):
        """Combine `score_sp` and `score_po`.

        `s`, `p` and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        The result is the horizontal concatenation of the outputs of ``score_sp(s,p)``
        and ``score_po(p,o)``. I.e., returns an :math:`n\\times 2E` tensor, where
        :math:`E` is the total number of known entities. For :math:$j<E$, the
        :math:`(i,j)`-entry holds the score for triple :math:`(s_i, p_i, j)`. For
        :math:$j\\ge E$, the :math:`(i,j)`-entry holds the score for triple :math:`(j-E,
        p_i, o_i)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            all_entities = self.get_s_embedder().embed_all()
            sp_scores = self.score_emb(s, p, all_entities, combine="sp*")
            po_scores = self.score_emb(all_entities, p, o, combine="*po")
        else:
            all_objects = self.get_o_embedder().embed_all()
            sp_scores = self.score_emb(s, p, all_objects, combine="sp*")
            all_subjects = self.get_s_embedder().embed_all()
            po_scores = self.score_emb(all_subjects, p, o, combine="*po")
        return torch.cat((sp_scores, po_scores), dim=1)


class KgeEmbedder(KgeBase):
    """
    Base class for all relational model embedders
    """

    def __init__(self, config, dataset, configuration_key):
        super().__init__(config, dataset)
        self.configuration_key = configuration_key
        self.embedder_type = config.get(configuration_key + ".type")

        # verify all custom options by trying to set them in a copy of this
        # configuration (quick and dirty, but works)
        custom_options = Config.flatten(config.get(configuration_key))
        del custom_options["type"]
        dummy_config = self.config.clone()
        for key, value in custom_options.items():
            try:
                dummy_config.set(self.embedder_type + "." + key, value)
            except ValueError:
                raise ValueError(
                    "key {}.{} invalid or of incorrect type".format(
                        self.configuration_key, key
                    )
                )

    @staticmethod
    def create(config, dataset, configuration_key, vocab_size):
        """Factory method for embedder creation."""

        embedder = None
        try:
            embedder_type = config.get(configuration_key + ".type")
            class_name = config.get(embedder_type + ".class_name")
            module = importlib.import_module("kge.model")
            embedder = getattr(module, class_name)(
                config, dataset, configuration_key, vocab_size
            )
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for embedder {}".format(
                    class_name, embedder_type
                )
            )

        return embedder

    def embed(self, indexes) -> torch.Tensor:
        """
        Computes the embedding.
        """
        raise NotImplementedError

    def embed_all(self) -> torch.Tensor:
        """
        Returns all embeddings.
        """
        raise NotImplementedError

        # TODO I/O

    def get_option(self, name):
        try:
            # custom option
            return self.config.get(self.configuration_key + "." + name)
        except KeyError:
            # default option
            return self.config.get(self.embedder_type + "." + name)
