import torch.optim
from torch.optim.lr_scheduler import _LRScheduler


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model):
        """ Factory method for optimizer creation """
        try:
            optimizer = getattr(torch.optim, config.get("train.optimizer"))
            return optimizer(
                [p for p in model.parameters() if p.requires_grad],
                **config.get("train.optimizer_args")
            )
        except:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.optimizer")


class KgeLRScheduler:
    """ Wraps torch learning rate (LR) schedulers """

    @staticmethod
    def create(config, optimizer):
        """ Factory method for LR scheduler creation """
        try:
            name = config.get("train.lr_scheduler")
            metric_based = False
            if name == 'ConstantLRScheduler':
                return ConstantLRScheduler(optimizer), metric_based
            args = config.get("train.lr_scheduler_args")
            scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **args)
            metric_based_schedulers = ['ReduceLROnPlateau']
            if name in metric_based_schedulers:
                metric_based = True
            return scheduler, metric_based
        except Exception as e:
            raise ValueError("Invalid LR scheduler options. Could not find '{}' "
                             "in torch.optim.lr_scheduler, error was {}".format(name, e))

class ConstantLRScheduler(_LRScheduler):
    """Default LR scheduler that does nothing."""

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
