from kge import Config, Configurable
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
import re
from operator import or_
from functools import reduce


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model):
        """ Factory method for optimizer creation """
        try:
            optimizer = getattr(torch.optim, config.get("train.optimizer"))
            return optimizer(
                KgeOptimizer._get_parameter_specific_options(config, model),
                **config.get("train.optimizer_args"),  # default optimizer options
            )
        except AttributeError:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(
                f"Could not create optimizer {config.get('train.optimizer')}. "
                f"Please specify an optimizer provided in torch.optim"
            )

    @staticmethod
    def _get_parameter_specific_options(config, model):
        named_parameters = dict(model.named_parameters())
        override_parameters = config.get("train.optimizer_args_override")
        parameter_names_per_search = dict()
        # filter named parameters by regex string
        for regex_string in override_parameters.keys():
            search_pattern = re.compile(regex_string)
            filtered_named_parameters = set(
                filter(search_pattern.match, named_parameters.keys())
            )
            parameter_names_per_search[regex_string] = filtered_named_parameters
        # check if something was matched by multiple strings
        parameter_values = list(parameter_names_per_search.values())
        for i, (regex_string, param) in enumerate(parameter_names_per_search.items()):
            for j in range(i + 1, len(parameter_names_per_search)):
                intersection = set.intersection(param, parameter_values[j])
                if len(intersection) > 0:
                    raise ValueError(
                        f"The parameters {intersection}, were matched by the override "
                        f"key {regex_string} and {list(parameter_names_per_search.keys())[j]}"
                    )
        # now we need to create a list like [{params: [parameters], options},..]
        for regex_string, params in parameter_names_per_search.items():
            override_parameters[regex_string]["params"] = [
                named_parameters[param] for param in params
            ]
        resulting_parameters = list(override_parameters.values())
        # we still need the unmatched parameters...
        default_parameter_names = set.difference(
            set(named_parameters.keys()),
            reduce(or_, list(parameter_names_per_search.values())),
        )
        resulting_parameters.extend(
            [
                {"params": named_parameters[default_parameter_name]}
                for default_parameter_name in default_parameter_names
            ]
        )
        return resulting_parameters


class KgeLRScheduler(Configurable):
    """ Wraps torch learning rate (LR) schedulers """

    def __init__(self, config: Config, optimizer):
        super().__init__(config)
        name = config.get("train.lr_scheduler")
        args = config.get("train.lr_scheduler_args")
        self._lr_scheduler: _LRScheduler = None
        if name != "":
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
