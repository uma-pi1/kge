from kge import Config
import copy
import torch.nn
import importlib


class KgeBase(torch.nn.Module):
    """
    Base class for all relational models and embedders
    """

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

    def initialize(self, what, initialize, initialize_arg):
        if initialize == 'normal':
            torch.nn.init.normal_(what, std=initialize_arg)
        else:
            raise ValueError("initialize")


class KgeModel(KgeBase):
    """
    Base class for all relational models
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # TODO generalize this
        self._entity_embedder = KgeEmbedder.create(
            config, dataset, config.get('model') + '.entity_embedder', True)
        self._relation_embedder = KgeEmbedder.create(
            config, dataset, config.get('model') + '.relation_embedder', False)

    def score_spo(self, s, p, o):
        return self._score(s, p, o)

    def score_sp(self, s, p):
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        all_objects = self.get_o_embedder().embed_all()
        return self._score(s, p, all_objects, prefix='sp')

    def score_po(self, p, o):
        all_subjects = self.get_s_embedder().embed_all()
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        return self._score(all_subjects, p, o, prefix='po')

    def score_sp_po(self, s, p, o):
        s = self.get_s_embedder().embed(s)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o)
        if self.get_s_embedder() is self.get_o_embedder():
            all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._score(s, p, all_entities, prefix='sp')
            po_scores = self._score(all_entities, p, o, prefix='po')
        else:
            all_objects = self.get_o_embedder().embed_all()
            sp_scores = self._score(s, p, all_objects, prefix='sp')
            all_subjects = self.get_s_embedder().embed_all()
            po_scores = self._score(all_subjects, p, o, prefix='po')
        return torch.cat((sp_scores, po_scores), dim=1)

    def score_p(self, p):
        raise NotImplementedError

    @staticmethod
    def create(config, dataset):
        """Factory method for model creation."""

        model = None
        try:
            model_name = config.get('model')
            class_name = config.get(model_name + '.class_name')
            module = importlib.import_module('kge.model')
            model = getattr(module, class_name)(config, dataset)
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("Can't find class {} in 'kge.model' for model {}".
                             format(class_name, model_name))

        # TODO I/O (resume model)
        model.to(config.get('job.device'))
        return model

    # TODO document this method and in particular: prefix
    def _score(self, s, p, o, prefix=None):
        r"""
        :param s: tensor of size [batch_size, embedding_size]
        :param p: tensor of size [batch_size, embedding_size]
        :param o:: tensor of size [batch_size, embedding_size]
        :return: score tensor of size [batch_size, 1]"""
        raise NotImplementedError

    def get_s_embedder(self):
        return self._entity_embedder

    def get_o_embedder(self):
        return self._entity_embedder

    def get_p_embedder(self):
        return self._relation_embedder


class KgeEmbedder(KgeBase):
    """
    Base class for all relational model embedders
    """

    def __init__(self, config, dataset, configuration_key, is_entity_embedder):
        super().__init__(config, dataset)
        self.configuration_key = configuration_key
        self.embedder_type = config.get(configuration_key + ".type")
        self.is_entity_embedder = is_entity_embedder

        # verify all custom options by trying to set them in a copy of this
        # configuration (quick and dirty, but works)
        custom_options = Config.flatten(config.get(configuration_key))
        del custom_options['type']
        dummy_config = copy.deepcopy(self.config)
        for key, value in custom_options.items():
            try:
                dummy_config.set(self.embedder_type + '.' + key, value)
            except ValueError:
                raise ValueError('key {}.{} invalid or of incorrect type'
                                 .format(self.configuration_key, key))

    @staticmethod
    def create(config, dataset, configuration_key, is_entity_embedder):
        """Factory method for embedder creation."""

        embedder = None
        try:
            embedder_type = config.get(configuration_key + ".type")
            class_name = config.get(embedder_type + '.class_name')
            module = importlib.import_module('kge.model')
            embedder = getattr(module, class_name)(
                config, dataset, configuration_key, is_entity_embedder)
        except ImportError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                "Can't find class {} in 'kge.model' for embedder {}"
                .format(class_name, embedder_type))


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
            return self.config.get(self.configuration_key + '.' + name)
        except KeyError:
            # default option
            return self.config.get(self.embedder_type + '.' + name)
