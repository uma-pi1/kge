import torch


class KgeLoss:
    """ Wraps torch loss functions """

    def create(config):
        """ Factory method for loss creation """
        if config.get('train.loss') == 'ce':
            return torch.nn.CrossEntropyLoss(reduction='average')
        elif config.get('train.loss') == 'bce':
            return torch.nn.BCEWithLogitsLoss(reduction='average')
        elif config.get('train.loss') == 'kl':
            return torch.nn.KLDivLoss(reduction='average')
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError('train.loss')
