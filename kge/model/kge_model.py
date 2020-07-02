import importlib
import tempfile
from collections import OrderedDict

from torch import Tensor
import torch.nn
import numpy as np
import os

import kge
from kge import Config, Configurable, Dataset
from kge.misc import filename_in_module
from kge.util import load_checkpoint
from typing import Any, Dict, List, Optional, Union, Tuple

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
        self.backward_compatible_keys = {
            "_entity_embedder.embeddings.weight": "_entity_embedder._embeddings.weight",
            "_relation_embedder.embeddings.weight": "_relation_embedder._embeddings.weight",
            "_base_model._entity_embedder.embeddings.weight": "_base_model._entity_embedder._embeddings.weight",
            "_base_model._relation_embedder.embeddings.weight": "_base_model._relation_embedder._embeddings.weight",
        }

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
        # handle deprecated keys
        state_dict = OrderedDict()

        for k, v in savepoint[0].items():
            state_dict[self.backward_compatible_keys.get(k, k)] = v

        self.load_state_dict(state_dict)
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
        values are :code:`"spo"`, :code:`"sp_"`, and :code:`"_po"`. Not all models may
        support all combinations.

        When `combine` is :code:`"spo"`, then embeddings are combined row-wise. In this
        case, it is required that :math:`n_s=n_p=n_o=n`. The output is identical to
        :func:`~RelationalScorer.score_emb_spo`, i.e., a :math`n\times 1` tensor, in
        which the :math:`i`-th entry holds the score of the embedding triple
        :math:`(s_i, p_i, o_i)`.

        When `combine` is :code:`"sp_"`, the subjects and predicates are taken row-wise
        and subsequently combined with all objects. In this case, it is required that
        :math:`n_s=n_p=n`. The output is a :math`n\times n_o` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple :math:`(s_i, p_i,
        o_j)`.

        When `combine` is :code:`"_po"`, predicates and objects are taken row-wise and
        subsequently combined with all subjects. In this case, it is required that
        :math:`n_p=n_o=n`. The output is a :math`n\times n_s` tensor, in which the
        :math:`(i,j)`-th entry holds the score of the embedding triple :math:`(s_j, p_i,
        o_i)`.

        """
        n = p_emb.size(0)

        if combine == "spo":
            assert s_emb.size(0) == n and o_emb.size(0) == n
            out = self.score_emb_spo(s_emb, p_emb, o_emb)
        elif combine == "sp_":
            assert s_emb.size(0) == n
            n_o = o_emb.size(0)
            s_embs = s_emb.repeat_interleave(n_o, 0)
            p_embs = p_emb.repeat_interleave(n_o, 0)
            o_embs = o_emb.repeat((n, 1))
            out = self.score_emb_spo(s_embs, p_embs, o_embs)
        elif combine == "_po":
            assert o_emb.size(0) == n
            n_s = s_emb.size(0)
            s_embs = s_emb.repeat((n, 1))
            p_embs = p_emb.repeat_interleave(n_s, 0)
            o_embs = o_emb.repeat_interleave(n_s, 0)
            out = self.score_emb_spo(s_embs, p_embs, o_embs)
        elif combine == "s_o":
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

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        init_for_load_only=False,
    ):
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

    def _init_embeddings(self, data: Tensor):
        """Initialize embeddings with provided configuration."""
        initialize = self.get_option("initialize")

        try:
            initialize_args_key = "initialize_args." + initialize
            initialize_args = self.get_option(initialize_args_key)
        except KeyError:
            initialize_args_key = "initialize_args"
            initialize_args = self.get_option(initialize_args_key)

        # Automatically set arg a (lower bound) for uniform_ if not given
        if initialize == "uniform_" and "a" not in initialize_args:
            initialize_args["a"] = initialize_args["b"] * -1
            self.set_option(initialize_args_key + ".a", initialize_args["a"], log=True)

        self.initialize(data, initialize, initialize_args)

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
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
                config,
                dataset,
                configuration_key,
                vocab_size,
                init_for_load_only=init_for_load_only,
            )
            return embedder
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for embedder {}".format(
                    class_name, embedder_type
                )
            )

    def _intersect_ids_with_pretrained_embedder(
        self, pretrained_embedder: "KgeEmbedder"
    ) -> Tuple[np.array, np.array]:
        """
        Intersect entity/relation ids of the embedder with embedderings of a pretrained
        embedder.
        Args:
            pretrained_embedder: KgeEmbedder with pre-trained embeddings

        Returns:
            self_intersection_ind: index if the intersecting entities/relations
                                   in this embedder
            pretrained_intersection_ind: index of intersecting entities/relations
                                         in the pretrained embedder
        """
        if "entity_embedder" in self.configuration_key:
            self_ids = self.dataset.entity_ids()
            pretrained_ids = pretrained_embedder.dataset.entity_ids()
        elif "relation_embedder" in self.configuration_key:
            self_ids = self.dataset.relation_ids()
            pretrained_ids = pretrained_embedder.dataset.relation_ids()
        else:
            raise ValueError(
                "Can only initialize entity or relation embedder with"
                " pretrained embeddings"
            )

        _, self_intersect_ind, pretrained_intersect_ind = np.intersect1d(
            self_ids, pretrained_ids, return_indices=True
        )
        if self.get_option("pretrain.ensure_all") and not len(
            self_intersect_ind
        ) == len(self_ids):
            raise IndexError(
                "Not all embeddings could be initialized with the embeddings provided "
                "in the pre-trained model"
            )
        return self_intersect_ind, pretrained_intersect_ind

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: "KgeEmbedder") -> None:
        """
        Initialize embedding layer with pre-trained embeddings from another embedder.
        Maps embeddings based on the entity/relation ids.
        Args:
            pretrained_embedder: KgeEmbedder with pre-trained embeddings

        Returns:
            None
        """
        raise NotImplementedError

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
        create_embedders=True,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(config, dataset, configuration_key)

        # TODO support different embedders for subjects and objects

        #: Embedder used for entities (both subject and objects)
        self._entity_embedder: KgeEmbedder

        #: Embedder used for relations
        self._relation_embedder: KgeEmbedder

        if create_embedders:
            self._entity_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".entity_embedder",
                dataset.num_entities(),
                init_for_load_only=init_for_load_only,
            )

            #: Embedder used for relations
            num_relations = dataset.num_relations()
            self._relation_embedder = KgeEmbedder.create(
                config,
                dataset,
                self.configuration_key + ".relation_embedder",
                num_relations,
                init_for_load_only=init_for_load_only,
            )

            if not init_for_load_only:
                # load pretrained embeddings
                pretrained_entities_filename = ""
                pretrained_relations_filename = ""
                if self.has_option("entity_embedder.pretrain.model_filename"):
                    pretrained_entities_filename = self.get_option(
                        "entity_embedder.pretrain.model_filename"
                    )
                if self.has_option("relation_embedder.pretrain.model_filename"):
                    pretrained_relations_filename = self.get_option(
                        "relation_embedder.pretrain.model_filename"
                    )

                def load_pretrained_model(
                    pretrained_filename: str,
                ) -> Optional[KgeModel]:
                    if pretrained_filename != "":
                        self.config.log(
                            f"Initializing with embeddings stored in "
                            f"{pretrained_filename}"
                        )
                        checkpoint = load_checkpoint(pretrained_filename)
                        return KgeModel.create_from(checkpoint)
                    return None

                pretrained_entities_model = load_pretrained_model(
                    pretrained_entities_filename
                )
                if pretrained_entities_filename == pretrained_relations_filename:
                    pretrained_relations_model = pretrained_entities_model
                else:
                    pretrained_relations_model = load_pretrained_model(
                        pretrained_relations_filename
                    )
                if pretrained_entities_model is not None:
                    if (
                        pretrained_entities_model.get_s_embedder()
                        != pretrained_entities_model.get_o_embedder()
                    ):
                        raise ValueError(
                            "Can only initialize with pre-trained models having "
                            "identical subject and object embeddings."
                        )
                    self._entity_embedder.init_pretrained(
                        pretrained_entities_model.get_s_embedder()
                    )
                if pretrained_relations_model is not None:
                    self._relation_embedder.init_pretrained(
                        pretrained_relations_model.get_p_embedder()
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
        config: Config,
        dataset: Dataset,
        configuration_key: Optional[str] = None,
        init_for_load_only=False,
    ) -> "KgeModel":
        """Factory method for model creation."""

        try:
            if configuration_key is not None:
                model_name = config.get(configuration_key + ".type")
            else:
                model_name = config.get("model")
            class_name = config.get(model_name + ".class_name")
            module = importlib.import_module("kge.model")
        except:
            raise Exception("Can't find {}.type in config".format(configuration_key))

        try:
            model = getattr(module, class_name)(
                config=config,
                dataset=dataset,
                configuration_key=configuration_key,
                init_for_load_only=init_for_load_only,
            )
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
        model: Optional[str] = None,
        dataset: Optional[Union[Dataset, str]] = None,
        options: Dict[str, Any] = {},
        folder: Optional[str] = None,
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
            dataset = Dataset.create(config)
        model = KgeModel.create(config, dataset)
        return model

    @staticmethod
    def create_from(
        checkpoint: Dict,
        dataset: Optional[Dataset] = None,
        use_tmp_log_folder=True,
        new_config: Config = None,
    ) -> "KgeModel":
        """Loads a model from a checkpoint file of a training job or a packaged model.

        If dataset is specified, associates this dataset with the model. Otherwise uses
        the dataset used to train the model.

        If `use_tmp_log_folder` is set, the logs and traces are written to a temporary
        file. Otherwise, the files `kge.log` and `trace.yaml` will be created (or
        appended to) in the checkpoint's folder.

        """
        config = Config.create_from(checkpoint)
        if new_config:
            config.load_config(new_config)

        if use_tmp_log_folder:
            import tempfile

            config.log_folder = tempfile.mkdtemp(prefix="kge-")
        else:
            config.log_folder = checkpoint["folder"]
            if not config.log_folder or not os.path.exists(config.log_folder):
                config.log_folder = "."
        dataset = Dataset.create_from(checkpoint, config, dataset, preload_data=False)
        model = KgeModel.create(config, dataset, init_for_load_only=True)
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
        # Note: If the subject and object embedder are identical, embeddings may be
        # penalized twice. This is intended (and necessary, e.g., if the penalty is
        # weighted).
        if "batch" in kwargs and "triples" in kwargs["batch"]:
            triples = kwargs["batch"]["triples"].to(self.config.get("job.device"))
            return (
                super().penalty(**kwargs)
                + self.get_s_embedder().penalty(indexes=triples[:, S], **kwargs)
                + self.get_p_embedder().penalty(indexes=triples[:, P], **kwargs)
                + self.get_o_embedder().penalty(indexes=triples[:, O], **kwargs)
            )
        else:
            return (
                super().penalty(**kwargs)
                + self.get_s_embedder().penalty(**kwargs)
                + self.get_p_embedder().penalty(**kwargs)
                + self.get_o_embedder().penalty(**kwargs)
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
        return self._scorer.score_emb(s, p, o, combine="spo").view(-1)

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
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

        return self._scorer.score_emb(s, p, o, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
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

        return self._scorer.score_emb(s, p, o, combine="_po")

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

        return self._scorer.score_emb(s, p, o, combine="s_o")

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
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
            po_scores = self._scorer.score_emb(all_entities, p, o, combine="_po")
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset)
                all_subjects = self.get_s_embedder().embed(entity_subset)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp_")
            po_scores = self._scorer.score_emb(all_subjects, p, o, combine="_po")
        return torch.cat((sp_scores, po_scores), dim=1)
