import torch


class KgeBase(torch.nn.Module):
    """
    Base class for all relational models and embedders
    """

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.is_cuda = False

    def cuda(self, device=None):
        super().cuda(device=device)
        self.is_cuda = True

    def cpu(self):
        super().cpu()
        self.is_cuda = False

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

    def score_spo(self, s, p, o):
        raise NotImplementedError

    def score_sp(self, s, p, is_training=False):
        raise NotImplementedError

    def score_po(self, p, o, is_training=False):
        raise NotImplementedError

    def score_p(self, p, is_training=False):
        raise NotImplementedError

    def create(config, dataset):
        """Factory method for model creation."""
        from kge.model import ComplEx

        ## create the embedders
        if config.get('model.type') == 'complex':
            return ComplEx(config, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError('model.type')

            # TODO I/O


class KgeEmbedder(KgeBase):
    """
    Base class for all relational model embedders
    """

    def __init__(self, config, dataset, is_entity_embedder):
        super().__init__(config, dataset)
        self.is_entity_embedder = is_entity_embedder

    def create(config, dataset, is_entity_embedder):
        """Factory method for embedder creation."""
        from kge.model import LookupEmbedder

        embedder_type = KgeEmbedder._get_option(config, 'model.embedder', is_entity_embedder)
        if embedder_type == 'lookup':
            return LookupEmbedder(config, dataset, is_entity_embedder)
        else:
            raise ValueError('embedder')

    def embed(self, indexes, is_training=False) -> torch.Tensor:
        """
        Computes the embedding.
        """
        raise NotImplementedError

    def embed_all(self, is_training=False) -> torch.Tensor:
        """
        Returns all embeddings.
        """
        raise NotImplementedError

        # TODO I/O

    def get_option(self, name):
        return KgeEmbedder._get_option(self.config, name, self.is_entity_embedder)

    def _get_option(config, name, is_entity_embedder):
        value = config.get(name)
        if type(value) == list:
            return value[0 if self.is_entity_embedder else 1]
        else:
            return value
