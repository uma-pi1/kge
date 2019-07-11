import torch.optim


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
            if not name:
                return name, metric_based
            args = config.get("train.lr_scheduler_args")
            scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **args)
            metric_based_schedulers = ['ReduceLROnPlateau']
            if name in metric_based_schedulers:
                metric_based = True
            return scheduler, metric_based
        except:
            raise ValueError("invalid LR scheduler options")
