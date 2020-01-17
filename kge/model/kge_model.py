import importlib
import tempfile

from torch import Tensor
import torch.nn
import os

import kge
from kge import Config, Configurable, Dataset
from kge.misc import filename_in_module
from typing import Any, Dict, List, Optional, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kge.job import Job


SLOTS = [0, 1, 2]
S, P, O = SLOTS


class KgeBase(torch.nn.Module, Configurable):
    r"""Base class for all KGE models, scorers, and embedders."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)
        self.dataset = dataset
        self.meta: Dict[str, Any] = dict()  #: meta-data stored with this module

    def initialize(self, what: Tensor, initialize: str, initialize_args):
        try:
            getattr(torch.nn.init, initialize)(what, **initialize_args)
        except:
            raise ValueError(
                "invalid initialization options: {} with args {}".format(
                    initialize, initialize_args
                )
            )

    def prepare_job(self, job: "Job", **kwargs):
        r"""Prepares the given job to work with this model.

        If this model does not support the specified job type, this function may raise
        an error.

        This function commonly registers hooks specific to this model. For a list of
        available hooks during training or evaluation, see :class:`TrainingJob` or
        :class:`EvaluationJob`:, respectively.

        """

    def penalty(self, **kwargs) -> List[Tensor]:
        r"""Returns additional penalty terms that are added to the loss during training.

        This method is called once per batch during training. The arguments being passed
        depend on the trainer being used.

        Returns a (possibly empty) list of penalty terms.

        """

        return []

    def save(self):
        "Returns data structure to save state"
        return (self.state_dict(), self.meta)

    def load(self, savepoint):
        "Loads state from a saved data structure"
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

    def __init__(self, config: Config, dataset: Dataset, configuration_key: str):
        super().__init__(config, dataset, configuration_key)

    def score_emb_spo(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor) -> Tensor:
        r"""Scores a set of triples specified by their embeddings.

        `s_emb`, `p_emb`, and `o_emb` are tensors of size :math:`n\times d_e`,
        :math:`n\times d_r`, and :math:`n\times d_e`, where :math:`d_e` and
        :math:`d_r` are the sizes of the entity and relation embeddings, respectively.

        The embeddings are combined row-wise. The output is a :math`n\times 1` tensor,
        in which the :math:`i`-th entry holds the score of the embedding triple
        :math:`(s_i, p_i, o_i)`.

        """
        return self.score_emb(s_emb, p_emb, o_emb, "spo")

    def score_emb(
        self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str
    ) -> Tensor:
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
        elif combine == "s*o":
            n = s_emb.size(0)
            assert o_emb.size(0) == n
            n_p = p_emb.size(0)
            s_embs = s_emb.repeat_interleave(n_p, 0)
            p_embs = p_emb.repeat((n, 1))
            o_embs = o_emb.repeat_interleave(n_p, 0)
            out = self.score_emb_spo(s_embs, p_embs, o_embs)
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class KgeEmbedder(KgeBase):
    r"""Base class for all embedders of a fixed number of objects.

    Objects can be entities, relations, mentions, and so on.

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key: str):
        super().__init__(config, dataset, configuration_key)

        #: location of the configuration options of this embedder
        self.embedder_type: str = self.get_option("type")

        # verify all custom options by trying to set them in a copy of this
        # configuration (quick and dirty, but works)
        try:
            custom_options = Config.flatten(config.get(self.configuration_key))
        except KeyError:
            # there are no custom options
            custom_options = {}
        if "type" in custom_options:
            del custom_options["type"]
        dummy_config = self.config.clone()
        for key, value in custom_options.items():
            try:
                dummy_config.set(self.embedder_type + "." + key, value)
            except ValueError as ve:
                raise ValueError(
                    "key {}.{} invalid or of incorrect type, message was {}".format(
                        self.configuration_key, key, ve
                    )
                )

        self.dim: int = self.get_option("dim")

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

    def forward(self, indexes: Tensor) -> Tensor:
        return self.embed(indexes)

    def embed(self, indexes: Tensor) -> Tensor:
        """Computes the embedding."""
        raise NotImplementedError

    def embed_all(self) -> Tensor:
        """Returns all embeddings."""
        raise NotImplementedError


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
        scorer: Union[RelationalScorer, type],
        initialize_embedders=True,
        configuration_key=None,
    ):
        super().__init__(config, dataset, configuration_key)

        # TODO support different embedders for subjects and objects

        #: Embedder used for entities (both subject and objects)
        self._entity_embedder: KgeEmbedder

        #: Embedder used for relations
        self._relation_embedder: KgeEmbedder

        if initialize_embedders:
            self._entity_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".entity_embedder",
                dataset.num_entities(),
            )

            #: Embedder used for relations
            num_relations = dataset.num_relations()
            self._relation_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".relation_embedder",
                num_relations,
            )

        #: Scorer
        self._scorer: RelationalScorer
        if type(scorer) == type:
            # scorer is type of the scorer to use; call its constructor
            self._scorer = scorer(
                config=config, dataset=dataset, configuration_key=self.configuration_key
            )
        else:
            self._scorer = scorer

    # overridden to also set self.model
    def _init_configuration(self, config: Config, configuration_key: Optional[str]):
        Configurable._init_configuration(self, config, configuration_key)
        if not hasattr(self, "model") or not self.model:
            if self.configuration_key:
                self.model: str = config.get(self.configuration_key + ".type")
            else:
                self.model: str = config.get("model")
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
    def create_default(
        model: "KgeModel" = None,
        dataset: Dataset = None,
        options: Dict[str, Any] = {},
        folder: str = None,
    ) -> "KgeModel":
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
    def load_from_checkpoint(
        filename: str, dataset=None, use_tmp_log_folder=True, device="cpu"
    ) -> "KgeModel":
        """Loads a model from a checkpoint file of a training job.

        If dataset is specified, associates this dataset with the model. Otherwise uses
        the dataset used to train the model.

        If `use_tmp_log_folder` is set, the logs and traces are written to a temporary
        file. Otherwise, the files `kge.log` and `trace.yaml` will be created (or
        appended to) in the checkpoint's folder.

        """

        checkpoint = torch.load(filename, map_location=device)

        original_config = checkpoint["config"]
        config = Config()  # round trip to handle deprecated configs
        config.load_options(original_config.options)
        config.set("job.device", device)
        if use_tmp_log_folder:
            import tempfile

            config.log_folder = tempfile.mkdtemp(prefix="kge-")
        else:
            config.log_folder = os.path.dirname(filename)
            if not config.log_folder:
                config.log_folder = "."
        if dataset is None:
            dataset = Dataset.load(config, preload_data=False)
        model = KgeModel.create(config, dataset)
        model.load(checkpoint["model"])
        model.eval()
        return model

    def prepare_job(self, job: "Job", **kwargs):
        super().prepare_job(job, **kwargs)
        self._entity_embedder.prepare_job(job, **kwargs)
        self._relation_embedder.prepare_job(job, **kwargs)

        def append_num_parameter(job, trace):
            trace["num_parameters"] = sum(map(lambda p: p.numel(), self.parameters()))

        job.post_epoch_trace_hooks.append(append_num_parameter)

    def penalty(self, **kwargs) -> List[Tensor]:
        if "batch" in kwargs and "triples" in kwargs["batch"]:
            kwargs["batch"]["triples"] = kwargs["batch"]["triples"].to(
                self.config.get("job.device")
            )
        return (
            # Note: If the subject and object embedder are identical, embeddings may be
            # penalized twice. This is intended (and necessary, e.g., if the penalty is
            # weighted).
            super().penalty(**kwargs)
            + self.get_s_embedder().penalty(slot=S, **kwargs)
            + self.get_p_embedder().penalty(slot=P, **kwargs)
            + self.get_o_embedder().penalty(slot=O, **kwargs)
        )

    def get_s_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_o_embedder(self) -> KgeEmbedder:
        return self._entity_embedder

    def get_p_embedder(self) -> KgeEmbedder:
        return self._relation_embedder

    def get_scorer(self) -> RelationalScorer:
        return self._scorer

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        r"""Compute scores for a set of triples.

        `s`, `p`, and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        `direction` may influence how scores are computed. For most models, this setting
        has no meaning. For reciprocal relations, direction must be either `"s"` or
        `"o"` (depending on what is predicted).

        Returns a vector of size :math:`n`, in which the :math:`i`-th entry holds the
        score of triple :math:`(s_i, p_i, o_i)`.

        """
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s, p, o, combine="spo")

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None, p_unique: Tensor = None) -> Tensor:
        r"""Compute scores for triples formed from a set of sp-pairs and all (or a subset of the) objects.

        `s` and `p` are vectors of common size :math:`n`, holding the indexes of the
        subjects and relations to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        p_i, j)`.

        If `o` is not None, it is a vector holding the indexes of the objects to score.

        If train.unique relations is true, every relation will only be embedded once
            p_unique is only needed in this setting and will be created if not provided

        """
        s_emb = self.get_s_embedder().embed(s)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        if self.config.get("train.unique_relations"):
            if p_unique is None:
                p_unique = torch.unique(p)
            p_unique_emb = self.get_p_embedder().embed(p_unique)
            scores = torch.empty([s_emb.size(0), o_emb.size(0)])
            for i in range(p_unique.size(0)):
                relation_index = p == p_unique[i]
                scores_chunk = self._scorer.score_emb(s_emb[relation_index], p_unique_emb[i], o_emb,
                                                      combine="sp*_unique")
                scores[relation_index] = scores_chunk
            return scores
        else:
            p_emb = self.get_p_embedder().embed(p)
            return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="sp*")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None, p_unique: Tensor = None) -> Tensor:
        r"""Compute scores for triples formed from a set of po-pairs and (or a subset of the) subjects.

        `p` and `o` are vectors of common size :math:`n`, holding the indexes of the
        relations and objects to score.

        Returns an :math:`n\times E` tensor, where :math:`E` is the total number of
        known entities. The :math:`(i,j)`-entry holds the score for triple :math:`(j,
        p_i, o_i)`.

        If `s` is not None, it is a vector holding the indexes of the objects to score.

        If train.unique relations is true, every relation will only be embedded once
            p_unique is only needed in this setting and will be created if not provided

        """
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)

        if self.config.get("train.unique_relations"):
            if p_unique is None:
                p_unique = torch.unique(p)
            p_unique_emb = self.get_p_embedder().embed(p_unique)
            scores = torch.empty([o_emb.size(0), s_emb.size(0)])
            for i in range(p_unique.size(0)):
                relation_index = p == p_unique[i]
                scores_chunk = self._scorer.score_emb(s_emb, p_unique_emb[i], o_emb[relation_index],
                                                      combine="*po_unique")
                scores[relation_index] = scores_chunk
            return scores
        else:
            p_emb = self.get_p_embedder().embed(p)
            return self._scorer.score_emb(s_emb, p_emb, o_emb, combine="*po")

    def score_so(self, s: Tensor, o: Tensor, p: Tensor = None) -> Tensor:
        r"""Compute scores for triples formed from a set of so-pairs and all (or a subset of the) relations.

        `s` and `o` are vectors of common size :math:`n`, holding the indexes of the
        subjects and objects to score.

        Returns an :math:`n\times R` tensor, where :math:`R` is the total number of
        known relations. The :math:`(i,j)`-entry holds the score for triple :math:`(s_i,
        j, o_i)`.

        If `p` is not None, it is a vector holding the indexes of the relations to score.

        """
        s = self.get_s_embedder().embed(s)
        o = self.get_o_embedder().embed(o)
        if p is None:
            p = self.get_p_embedder().embed_all()
        else:
            p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o, combine="s*o")

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        r"""Combine `score_sp` and `score_po`.

        `s`, `p` and `o` are vectors of common size :math:`n`, holding the indexes of
        the subjects, relations, and objects to score.

        Each sp-pair and each po-pair is scored against the entities in `entity_subset`
        (also holds indexes). If set to `entity_subset` is `None`, scores against all
        entities.

        The result is the horizontal concatenation of the outputs of
        :code:`score_sp(s,p,entity_subset)` and :code:`score_po(p,o,entity_subset)`.
        I.e., returns an :math:`n\times 2E` tensor, where :math:`E` is the size of
        `entity_subset`. For :math:`j<E`, the :math:`(i,j)`-entry holds the score for
        triple :math:`(s_i, p_i, e_j)`. For :math:`j\ge E`, the :math:`(i,j)`-entry
        holds the score for triple :math:`(e_{j-E}, p_i, o_i)`.

        """

        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp*")
            po_scores = self._scorer.score_emb(all_entities, p, o, combine="*po")
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp*")
            po_scores = self._scorer.score_emb(all_subjects, p, o, combine="*po")
        return torch.cat((sp_scores, po_scores), dim=1)
