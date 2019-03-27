import torch.optim


class KgeOptimizer:
    """ Wraps torch optimizers """

    def create(config, model):
        """ Factory method for optimizer creation """
        if config.get('train.optimizer') == 'adagrad':
            return torch.optim.AdaGrad(model.parameters(), config.get('train.lr'))
        elif config.get('train.optimizer') == 'adam':
            if config.get('model.sparse'):
                return torch.optim.SparseAdam(model.parameters(), config.get('train.lr'))
            else:
                return torch.optim.Adam(model.parameters(), config.get('train.lr'))
        elif config.get('train.optimizer') == 'sgd':
            return torch.optim.SGD(model.parameters(), config.get('train.lr'))
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError('train.optimizer')

    # TODO: make class a proper wrapper, include scheduling
