import importlib
import tempfile

import torch.nn

import kge
from kge import Config, Dataset
from kge.util.misc import filename_in_module


class KgeBase(torch.nn.Module):
    r"""Base class for all KGE models, scorers, and embedders."""

    def __init__(self, config: Config, dataset: Dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.meta = dict()  #: meta-data stored with this module

    def initialize(self, what, initialize: str, initialize_args):
        try:
            getattr(torch.nn.init, initialize)(what, **initialize_args)
        except:
            if initialize == "auto_initialization":
                raise ValueError("{} does not support auto initialization.".format(self.config.get('model')))
            else:
                raise ValueError("invalid initialization options")

    def prepare_job(self, job, **kwargs):
        r"""Prepares the given job to work with this model.

        If this model does not support the specified job type, this function may raise
        an error.

        This function commonly registers hooks specific to this model. For a list of
        available hooks during training or evaluation, see :class:`TrainingJob` or
        :class:`EvaluationJob`:, respectively.

        """
        pass

    def penalty(self, **kwargs):
        r"""Returns additional penalty terms that are added to the loss during training.

        This method is called once per batch during training. The arguments being passed
        depend on the trainer being used.

        Returns a (possibly empty) list of penalty terms.

        """

        return []

    def save(self):
        "Returns data structure to save module state"
        return (self.state_dict(), self.meta)

    def load(self, savepoint):
        "Loads modulde state from a saved data structre"
        self.load_state_dict(savepoint[0])
        self.meta = savepoint[1]


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
        self.embedder_type = self.get_option("type")

        # verify all custom options by trying to set them in a copy of this
        # configuration (quick and dirty, but works)
        custom_options = Config.flatten(config.get(self.configuration_key))
        if "type" in custom_options:
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
            embedder_type = config.get_default(configuration_key + ".type")
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

    def forward(self, indexes):
        return self.embed(indexes)

    def embed(self, indexes):
        """Computes the embedding."""
        raise NotImplementedError

    def embed_all(self):
        """Returns all embeddings."""
        raise NotImplementedError

    def get_option(self, name):
        return self.config.get_default(self.configuration_key + "." + name)

    def check_option(self, name, allowed_values):
        return self.config.check_default(
            self.configuration_key + "." + name, allowed_values
        )


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
        configuration_key=None,
    ):
        super().__init__(config, dataset)
        self._init_configuration(config, configuration_key)

        # TODO support different embedders for subjects and objects

        #: Embedder used for entities (both subject and objects)
        self._entity_embedder = None

        #: Embedder used for relations
        self._relation_embedder = None

        if initialize_embedders:
            self._entity_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".entity_embedder",
                dataset.num_entities,
            )

            #: Embedder used for relations
            num_relations = dataset.num_relations
            self._relation_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".relation_embedder",
                num_relations,
            )

        #: Scorer
        self._scorer = scorer

    def _init_configuration(self, config, configuration_key):
        r"""Initializes self.config and self.configuration_key.

        Only after this method has been called, `get_option`, `check_option`, and
        `set_option` should be used. This method is automatically called in the
        constructor of this class, but can also be called by subclasses before calling
        the superclass constructor to allow access to these three methods.

        """
        self.config = config
        self.configuration_key = configuration_key
        if self.configuration_key:
            self.model = config.get(self.configuration_key + ".type")
        else:
            self.model = config.get("model")
            self.configuration_key = self.model

    @staticmethod
    def create(
        config: Config, dataset: Dataset, configuration_key: str = None
    ) -> "KgeModel":
        """Factory method for model creation."""

        try:
            if configuration_key:
                model_name = config.get(configuration_key + ".type")
            else:
                model_name = config.get("model")
            class_name = config.get(model_name + ".class_name")
            module = importlib.import_module("kge.model")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            model = getattr(module, class_name)(config, dataset, configuration_key)
            model.to(config.get("job.device"))
            return model
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for model {}".format(
                    class_name, model_name
                )
            )

    @staticmethod
    def create_all(model=None, dataset=None, options={}, folder=None):
        """Utility method to create a model, including configuration and dataset.

        `model` is the name of the model (takes precedence over
        ``options["model"]``), `dataset` a dataset name or `Dataset` instance (takes
        precedence over ``options["dataset.name"]``), and options arbitrary other
        configuration options.

        If `folder` is ``None``, creates a temporary folder. Otherwise uses the
        specified folder.

        """
        # load default model config
        if model is None:
            model = options["model"]
        default_config_file = filename_in_module(kge.model, "{}.yaml".format(model))
        config = Config()
        config.load(default_config_file, create=True)

        # apply specified options
        config.set("model", model)
        if isinstance(dataset, Dataset):
            config.set("dataset.name", dataset.config.get("dataset.name"))
        elif isinstance(dataset, str):
            config.set("dataset.name", dataset)
        config.set_all(new_options=options)

        # create output folder
        if folder is None:
            config.folder = tempfile.mkdtemp(
                "{}-{}-".format(config.get("dataset.name"), config.get("model"))
            )
        else:
            config.folder = folder

        # create dataset and model
        if not isinstance(dataset, Dataset):
            dataset = Dataset.load(config)
        model = KgeModel.create(config, dataset)
        return model

    @staticmethod
    def load_from_checkpoint(filename, dataset=None):
        """Loads a model from a checkpoint file of a training job.

        If dataset is specified, associates this dataset with the model. Otherwise uses
        the dataset used to train the model.

        """

        checkpoint = torch.load(filename)
        config = checkpoint["config"]
        if dataset is None:
            dataset = Dataset.load(config)
        model = KgeModel.create(config, dataset)
        model.load(checkpoint["model"])
        return model

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)
        self._entity_embedder.prepare_job(job, **kwargs)
        self._relation_embedder.prepare_job(job, **kwargs)

        def append_num_parameter(job, trace):
            trace["num_parameters"] = sum(map(lambda p: p.numel(), self.parameters()))

        job.post_epoch_trace_hooks.append(append_num_parameter)

    def penalty(self, **kwargs):
        return (
            super().penalty(**kwargs)
            + self._entity_embedder.penalty(**kwargs)
            + self._relation_embedder.penalty(**kwargs)
        )

    def get_s_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_o_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_p_embedder(self) -> KgeEmbedder:
        return self._relation_embedder

    def get_scorer(self) -> RelationalScorer:
        return self._scorer

    def set_option(self, name, value):
        if self.configuration_key:
            self.config.set(self.configuration_key + "." + name, value)
        else:
            self.config.set(name, value)

    def get_option(self, name):
        if self.configuration_key:
            return self.config.get_default(self.configuration_key + "." + name)
        else:
            return self.config.get_default(name)

    def check_option(self, name, allowed_values):
        if self.configuration_key:
            return self.config.check_default(self.configuration_key + "." + name)
        else:
            return self.config.check_default(name)

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

    def score_sp(self, s, p, o=None):
        r"""Compute scores for triples formed from a set of sp-pairs and all (or a subset of the) objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        If `o` is not None, it is a vector holding the indexes of the objects to score.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)

        return self._scorer.score_emb(s, p, o, combine="sp*")

    def score_po(self, p, o, s=None):
        r"""Compute scores for triples formed from a set of po-pairs and (or a subset of the) subjects.

        `p` and `o` are vectors of common size :math:`n`, holding the indexes of the
        relations and objects to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(j,
        p_i, o_i)`.

        If `s` is not None, it is a vector holding the indexes of the objects to score.

        """

        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
        o = self.get_o_embedder().embed(o)
        p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o, combine="*po")

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
