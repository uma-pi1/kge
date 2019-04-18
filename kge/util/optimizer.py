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

    # TODO: make class a proper wrapper, include scheduling
