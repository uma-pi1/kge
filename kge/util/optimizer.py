from kge import Config, Configurable
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model):
        """ Factory method for optimizer creation """
        try:
            optimizer = getattr(torch.optim, config.get("train.optimizer"))
            relation_parameters = set()
            relation_optimizer = None
            if config.get("train.relation_optimizer"):
                relation_optimizer = getattr(
                    torch.optim, config.get("train.relation_optimizer")
                )
                relation_parameters = set(
                    p for p in model.get_p_embedder().parameters() if p.requires_grad
                )
                relation_optimizer = relation_optimizer(
                    relation_parameters, **config.get("train.relation_optimizer_args")
                )
            parameters = [
                p
                for p in model.parameters()
                if p.requires_grad and p not in relation_parameters
            ]
            optimizer = optimizer(parameters, **config.get("train.optimizer_args"))
            return optimizer, relation_optimizer
        except AttributeError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                f"Could not create optimizer {config.get('train.optimizer')}. "
                f"Please specify an optimizer provided in torch.optim"
            )


class KgeLRScheduler(Configurable):
    """ Wraps torch learning rate (LR) schedulers """

    def __init__(self, config: Config, optimizer):
        super().__init__(config)
        name = config.get("train.lr_scheduler")
        args = config.get("train.lr_scheduler_args")
        self._lr_scheduler: _LRScheduler = None
        if name != "" and optimizer is not None:
            # check for consistency of metric-based scheduler
            self._metric_based = name in ["ReduceLROnPlateau"]
            if self._metric_based:
                desired_mode = "max" if config.get("valid.metric_max") else "min"
                if "mode" in args:
                    if args["mode"] != desired_mode:
                        raise ValueError(
                            (
                                "valid.metric_max ({}) and train.lr_scheduler_args.mode "
                                "({}) are inconsistent."
                            ).format(config.get("valid.metric_max"), args["mode"])
                        )
                    # all fine
                else:  # mode not set, so set it
                    args["mode"] = desired_mode
                    config.set("train.lr_scheduler_args.mode", desired_mode, log=True)

            # create the scheduler
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
                    optimizer, **args
                )
            except Exception as e:
                raise ValueError(
                    (
                        "Invalid LR scheduler {} or scheduler arguments {}. "
                        "Error: {}"
                    ).format(name, args, e)
                )

    def step(self, metric=None):
        if self._lr_scheduler is None:
            return
        if self._metric_based:
            if metric is not None:
                # metric is set only after validation has been performed, so here we
                # step
                self._lr_scheduler.step(metrics=metric)
        else:
            # otherwise, step after every epoch
            self._lr_scheduler.step()

    def state_dict(self):
        if self._lr_scheduler is None:
            return dict()
        else:
            return self._lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if self._lr_scheduler is None:
            pass
        else:
            self._lr_scheduler.load_state_dict(state_dict)
