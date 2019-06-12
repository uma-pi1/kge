from kge import Config, Dataset
import torch.nn
import importlib


class KgeBase(torch.nn.Module):
    r"""Base class for all KGE models, scorers, and embedders."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

    def initialize(self, what, initialize: str, initialize_arg):
        if initialize == "normal":
            torch.nn.init.normal_(what, std=initialize_arg)
        else:
            raise ValueError("initialize")


class RelationalScorer(KgeBase):
    r"""Base class for all relational scorers.

    Relational scorers take as input the embeddings of (subject, predicate,
    object)-triple and produce a score.

    Implementations of this class should either implement
    :func:`~RelationalScorer.score_emb_spo` (the quick way, but potentially inefficient)
    or :func:`~RelationalScorer.score_emb` (the hard way, potentially more efficient).

    """

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__(config, dataset)

    def score_emb_spo(self, s_emb, p_emb, o_emb):
        r"""Scores a set of triples specified by their embeddings.

        `s_emb`, `p_emb`, and `o_emb` are tensors of size :math:`n\times d_e`,
        :math:`n\times d_r`, and :math:`n\times d_e`, where :math:`d_e` and
        :math:`d_r` are the sizes of the entity and relation embeddings, respectively.

        The embeddings are combined row-wise. The output is a :math`n\times 1` tensor,
        in which the :math:`i`-th entry holds the score of the embedding triple
        :math:`(s_i, p_i, o_i)`.

        """
        return self.score_emb(s_emb, p_emb, o_emb, "spo")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        r"""Scores a set of triples specified by their embeddings.

        `s_emb`, `p_emb`, and `o_emb` are tensors of size :math:`n_s\times d_e`,
        :math:`n_p\times d_r`, and :math:`n_o\times d_e`, where :math:`d_e` and
        :math:`d_r` are the sizes of the entity and relation embeddings, respectively.

        The provided embeddings are combined based on the value of `combine`. Common
        values are :code:`"spo"`, :code:`"sp*"`, and :code:`"*po"`. Not all models may
        support all combinations.

        When `combine` is :code:`"spo"`, then embeddings are combined row-wise. In this
        case, it is required that :math:`n_s=n_p=n_o=n`. The output is identical to
        :func:`~RelationalScorer.score_emb_spo`, i.e., a :math`n\times 1` tensor, in
        which the :math:`i`-th entry holds the score of the embedding triple
        :math:`(s_i, p_i, o_i)`.

        When `combine` is :code:`"sp*"`, the subjects and predicates are taken row-wise
        and subsequently combined with all objects. In this case, it is required that
        :math:`n_s=n_p=n`. The output is a :math`n\times n_o` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple :math:`(s_i, p_i,
        o_j)`.

        When `combine` is :code:`"*po"`, predicates and objects are taken row-wise and
        subsequently combined with all subjects. In this case, it is required that
        :math:`n_p=n_o=n`. The output is a :math`n\times n_s` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple :math:`(s_j, p_i,
        o_i)`.

        """
        n = p_emb.size(0)

        if combine == "spo":
            assert s_emb.size(0) == n and o_emb.size(0) == n
            out = self.score_emb_spo(s_emb, p_emb, o_emb)
        elif combine == "sp*":
            assert s_emb.size(0) == n
            n_o = o_emb.size(0)
            s_embs = s_emb.repeat_interleave(n_o, 0)
            p_embs = p_emb.repeat_interleave(n_o, 0)
            o_embs = o_emb.repeat((n, 1))
            out = self.score_emb_spo(s_embs, p_embs, o_embs)
        elif combine == "*po":
            assert o_emb.size(0) == n
            n_s = s_emb.size(0)
            s_embs = s_emb.repeat((n, 1))
            p_embs = p_emb.repeat_interleave(n_s, 0)
            o_embs = o_emb.repeat_interleave(n_s, 0)
            out = self.score_emb_spo(s_embs, p_embs, o_embs)
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class KgeEmbedder(KgeBase):
    r"""Base class for all embedders of a fixed number of objects.

    Objects can be entities, relations, mentions, and so on.

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key: str):
        super().__init__(config, dataset)

        #: location of the configuration options of this embedder
        self.configuration_key = configuration_key

        #: type of this embedder
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

        self.dim = self.get_option("dim")

    @staticmethod
    def create(
        config: Config, dataset: Dataset, configuration_key: str, vocab_size: int
    ) -> "KgeEmbedder":
        """Factory method for embedder creation."""

        try:
            embedder_type = config.get(configuration_key + ".type")
            class_name = config.get(embedder_type + ".class_name")
            module = importlib.import_module("kge.model")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            embedder = getattr(module, class_name)(
                config, dataset, configuration_key, vocab_size
            )
            return embedder
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for embedder {}".format(
                    class_name, embedder_type
                )
            )

    def embed(self, indexes):
        """Computes the embedding."""
        raise NotImplementedError

    def embed_all(self):
        """Returns all embeddings."""
        raise NotImplementedError

    def get_option(self, name):
        try:
            # custom option
            return self.config.get(self.configuration_key + "." + name)
        except KeyError:
            # default option
            return self.config.get(self.embedder_type + "." + name)


class KgeModel(KgeBase):
    r"""Generic KGE model for KBs with a fixed set of entities and relations.

    This class uses :class:`KgeEmbedder` to associate each subject, relation, and object
    with an embedding, and a :class:`RelationalScorer` to score (subject, predicate,
    object) triples.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        scorer: RelationalScorer,
        initialize_embedders=True,
    ):
        super().__init__(config, dataset)

        # TODO support different embedders for subjects and objects

        #: Embedder used for entitites (both subject and objects)
        self._entity_embedder = None

        #: Embedder used for relations
        self._relation_embedder = None

        if initialize_embedders:
            self._entity_embedder = KgeEmbedder.create(
                config,
                dataset,
                config.get("model") + ".entity_embedder",
                dataset.num_entities,
            )

            #: Embedder used for relations
            self._relation_embedder = KgeEmbedder.create(
                config,
                dataset,
                config.get("model") + ".relation_embedder",
                dataset.num_relations,
            )

        #: Scorer
        self._scorer = scorer

    @staticmethod
    def create(config: Config, dataset: Dataset) -> "KgeModel":
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

    def get_s_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_o_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_p_embedder(self) -> KgeEmbedder:
        return self._relation_embedder

    def get_scorer(self) -> RelationalScorer:
        return self._scorer

    def score_spo(self, s, p, o):
        r"""Compute scores for a set of triples.

        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, combine="spo")

    def score_sp(self, s, p):
        r"""Compute scores for triples formed from a set of sp-pairs and all objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        all_objects = self.get_o_embedder().embed_all()
        return self._scorer.score_emb(s, p, all_objects, combine="sp*")

    def score_po(self, p, o):
        r"""Compute scores for triples formed from a set of po-pairs and all subjects.

        `p` and `o` are vectors of common size :math:`n`, holding the indexes of the
        relations and objects to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(j,
        p_i, o_i)`.

        """
        all_subjects = self.get_s_embedder().embed_all()
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(all_subjects, p, o, combine="*po")

    def score_sp_po(self, s, p, o):
        r"""Combine `score_sp` and `score_po`.

        `s`, `p` and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        The result is the horizontal concatenation of the outputs of
        :code:`score_sp(s,p)` and :code:`score_po(p,o)`. I.e., returns an :math:`n\times
        2E` tensor, where :math:`E` is the total number of known entities. For
        :math:`j<E`, the :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`. For :math:`j\ge E`, the :math:`(i,j)`-entry holds the score for triple
        :math:`(j-E, p_i, o_i)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp*")
            po_scores = self._scorer.score_emb(all_entities, p, o, combine="*po")
        else:
            all_objects = self.get_o_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp*")
            all_subjects = self.get_s_embedder().embed_all()
            po_scores = self._scorer.score_emb(all_subjects, p, o, combine="*po")
        return torch.cat((sp_scores, po_scores), dim=1)

    def post_score_loss_hook(self, epoch, epoch_step):
        r"""Return additional penalties that are can be added to the loss"""
        pass

    def post_update_trace_hook(self, trace_msg):
        r"""Add additional messages after the update"""
        return trace_msg

    def post_epoch_trace_hook(self, trace_msg):
        r"""Add additional messages after the epoch"""
        return trace_msg
