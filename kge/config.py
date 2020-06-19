from __future__ import annotations

import collections
import copy
import datetime
import os
import time
import uuid
import sys
from enum import Enum

import yaml
from typing import Any, Dict, List, Optional, Union


class Config:
    """Configuration options.

    All available options, their types, and their descriptions are defined in
    :file:`config_default.yaml`.
    """

    def __init__(self, folder: Optional[str] = None, load_default=True):
        """Initialize with the default configuration"""
        if load_default:
            import kge
            from kge.misc import filename_in_module

            with open(filename_in_module(kge, "config-default.yaml"), "r") as file:
                self.options: Dict[str, Any] = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            self.options = {}

        self.folder = folder  # main folder (config file, checkpoints, ...)
        self.log_folder: Optional[str] = (
            None  # None means use self.folder; used for kge.log, trace.yaml
        )
        self.log_prefix: str = None

    # -- ACCESS METHODS ----------------------------------------------------------------

    def get(self, key: str, remove_plusplusplus=True) -> Any:
        """Obtain value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type"). Strips all
        '+++' keys unless `remove_plusplusplus` is set to `False`.

        """
        result = self.options
        for name in key.split("."):
            try:
                result = result[name]
            except KeyError:
                raise KeyError(f"Error accessing {name} for key {key}")

        if remove_plusplusplus and isinstance(result, collections.Mapping):

            def do_remove_plusplusplus(option):
                if isinstance(option, collections.Mapping):
                    option.pop("+++", None)
                    for values in option.values():
                        do_remove_plusplusplus(values)

            result = copy.deepcopy(result)
            do_remove_plusplusplus(result)

        return result

    def get_default(self, key: str) -> Any:
        """Returns the value of the key if present or default if not.

        The default value is looked up as follows. If the key has form ``parent.field``,
        see if there is a ``parent.type`` property. If so, try to look up ``field``
        under the key specified there (proceeds recursively). If not, go up until a
        `type` field is found, and then continue from there.

        """
        try:
            return self.get(key)
        except KeyError as e:
            last_dot_index = key.rfind(".")
            if last_dot_index < 0:
                raise e
            parent = key[:last_dot_index]
            field = key[last_dot_index + 1 :]
            while True:
                # self.log("Looking up {}/{}".format(parent, field))
                try:
                    parent_type = self.get(parent + "." + "type")
                    # found a type -> go to this type and lookup there
                    new_key = parent_type + "." + field
                    last_dot_index = new_key.rfind(".")
                    parent = new_key[:last_dot_index]
                    field = new_key[last_dot_index + 1 :]
                except KeyError:
                    # no type found -> go up hierarchy
                    last_dot_index = parent.rfind(".")
                    if last_dot_index < 0:
                        raise e
                    field = parent[last_dot_index + 1 :] + "." + field
                    parent = parent[:last_dot_index]
                    continue
                try:
                    value = self.get(parent + "." + field)
                    # uncomment this to see where defaults are taken from
                    # self.log(
                    #     "Using value of {}={} for key {}".format(
                    #         parent + "." + field, value, key
                    #     )
                    # )
                    return value
                except KeyError:
                    # try further
                    continue

    def get_first_present_key(self, *keys: str, use_get_default=False) -> str:
        "Return the first key for which ``get`` or ``get_default`` finds a value."
        for key in keys:
            try:
                self.get_default(key) if use_get_default else self.get(key)
                return key
            except KeyError:
                pass
        raise KeyError("None of the following keys found: ".format(keys))

    def get_first(self, *keys: str, use_get_default=False) -> Any:
        "Return value (or default value) of the first valid key present or KeyError."
        if use_get_default:
            return self.get_default(
                self.get_first_present_key(*keys, use_get_default=True)
            )
        else:
            return self.get(self.get_first_present_key(*keys))

    def exists(self, key: str, remove_plusplusplus=True) -> bool:
        try:
            self.get(key, remove_plusplusplus)
            return True
        except KeyError:
            return False

    Overwrite = Enum("Overwrite", "Yes No Error")

    def set(
        self, key: str, value, create=False, overwrite=Overwrite.Yes, log=False
    ) -> Any:

        """Set value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type").

        If ``create`` is ``False`` , raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.

        """
        from kge.misc import is_number

        splits = key.split(".")
        data = self.options

        # flatten path and see if it is valid to be set
        path = []
        for i in range(len(splits) - 1):
            if splits[i] in data:
                create = create or "+++" in data[splits[i]]
            else:
                if create:
                    data[splits[i]] = dict()
                else:
                    msg = (
                        "Key '{}' cannot be set because key '{}' does not exist "
                        "and no new keys are allowed to be created "
                    ).format(key, ".".join(splits[: (i + 1)]))
                    if i == 0:
                        raise KeyError(msg + "at root level.")
                    else:
                        raise KeyError(
                            msg + "under key '{}'.".format(".".join(splits[:i]))
                        )

            path.append(splits[i])
            data = data[splits[i]]

        # check correctness of value
        try:
            current_value = data.get(splits[-1])
        except:
            raise Exception(
                "These config entries {} {} caused an error.".format(data, splits[-1])
            )

        if current_value is None:
            if not create:
                msg = (
                    f"Key '{key}' cannot be set because it does not exist and "
                    "no new keys are allowed to be created "
                )
                if len(path) == 0:
                    raise KeyError(msg + "at root level.")
                else:
                    raise KeyError(msg + ("under key '{}'.").format(".".join(path)))

            if isinstance(value, str) and is_number(value, int):
                value = int(value)
            elif isinstance(value, str) and is_number(value, float):
                value = float(value)
        else:
            if (
                isinstance(value, str)
                and isinstance(current_value, float)
                and is_number(value, float)
            ):
                value = float(value)
            elif (
                isinstance(value, str)
                and isinstance(current_value, int)
                and is_number(value, int)
            ):
                value = int(value)
            if type(value) != type(current_value):
                raise ValueError(
                    "key '{}' has incorrect type (expected {}, found {})".format(
                        key, type(current_value), type(value)
                    )
                )
            if overwrite == Config.Overwrite.No:
                return current_value
            if overwrite == Config.Overwrite.Error and value != current_value:
                raise ValueError("key '{}' cannot be overwritten".format(key))

        # all fine, set value
        data[splits[-1]] = value
        if log:
            self.log("Set {}={}".format(key, value))
        return value

    def _import(self, module_name: str):
        """Imports the specified module configuration.

        Adds the configuration options from kge/model/<module_name>.yaml to
        the configuration. Retains existing module configurations, but verifies
        that fields and their types are correct.

        """
        import kge.model, kge.model.embedder
        from kge.misc import filename_in_module

        # load the module_name
        module_config = Config(load_default=False)
        module_config.load(
            filename_in_module(
                [kge.model, kge.model.embedder], "{}.yaml".format(module_name)
            ),
            create=True,
        )
        if "import" in module_config.options:
            del module_config.options["import"]

        # add/verify current configuration
        for key in module_config.options.keys():
            cur_value = None
            try:
                cur_value = {key: self.get(key)}
            except KeyError:
                continue
            module_config.set_all(cur_value, create=False)

        # now update this configuration
        self.set_all(module_config.options, create=True)

        # remember the import
        imports = self.options.get("import")
        if imports is None:
            imports = module_name
        elif isinstance(imports, str):
            imports = [imports, module_name]
        else:
            imports.append(module_name)
            imports = list(dict.fromkeys(imports))
        self.options["import"] = imports

    def set_all(
        self, new_options: Dict[str, Any], create=False, overwrite=Overwrite.Yes
    ):
        for key, value in Config.flatten(new_options).items():
            self.set(key, value, create, overwrite)

    def load(
        self,
        filename: str,
        create=False,
        overwrite=Overwrite.Yes,
        allow_deprecated=True,
    ):
        """Update configuration options from the specified YAML file.

        All options that do not occur in the specified file are retained.

        If ``create`` is ``False``, raises :class:`ValueError` when the file
        contains a non-existing options. When ``create`` is ``True``, allows
        to add options that are not present in this configuration.

        If the file has an import or model field, the corresponding
        configuration files are imported.

        """
        with open(filename, "r") as file:
            new_options = yaml.load(file, Loader=yaml.SafeLoader)
        if new_options is not None:
            self.load_options(
                new_options,
                create=create,
                overwrite=overwrite,
                allow_deprecated=allow_deprecated,
            )

    def load_options(
        self, new_options, create=False, overwrite=Overwrite.Yes, allow_deprecated=True
    ):
        "Like `load`, but loads from an options object obtained from `yaml.load`."
        # import model configurations
        if "model" in new_options:
            model = new_options.get("model")
            # TODO not sure why this can be empty when resuming an ax
            # search with model as a search parameter
            if model:
                self._import(model)
        if "import" in new_options:
            imports = new_options.get("import")
            if not isinstance(imports, list):
                imports = [imports]
            for module_name in imports:
                self._import(module_name)
            del new_options["import"]

        # process deprecated options
        if allow_deprecated:
            new_options = _process_deprecated_options(Config.flatten(new_options))

        # now set all options
        self.set_all(new_options, create, overwrite)

    def load_config(
        self, config, create=False, overwrite=Overwrite.Yes, allow_deprecated=True
    ):
        "Like `load`, but loads from a Config object."
        self.load_options(config.options, create, overwrite, allow_deprecated)

    def save(self, filename):
        """Save this configuration to the given file"""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds the config file to a checkpoint"""
        checkpoint["config"] = self
        return checkpoint

    @staticmethod
    def flatten(options: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a dictionary of flattened configuration options."""
        result = {}
        Config.__flatten(options, result)
        return result

    @staticmethod
    def __flatten(options: Dict[str, Any], result: Dict[str, Any], prefix=""):
        for key, value in options.items():
            fullkey = key if prefix == "" else prefix + "." + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    def clone(self, subfolder: str = None) -> "Config":
        """Return a deep copy"""
        new_config = Config(folder=copy.deepcopy(self.folder), load_default=False)
        new_config.options = copy.deepcopy(self.options)
        if subfolder is not None:
            new_config.folder = os.path.join(self.folder, subfolder)
        return new_config

    # -- LOGGING AND TRACING -----------------------------------------------------------

    def log(self, msg: str, echo=True, prefix=""):
        """Add a message to the default log file.

        Optionally also print on console. ``prefix`` is used to indent each
        output line.

        """
        with open(self.logfile(), "a") as file:
            for line in msg.splitlines():
                if prefix:
                    line = prefix + line
                if self.log_prefix:
                    line = self.log_prefix + line
                if echo:
                    self.print(line)
                file.write(str(datetime.datetime.now()) + " " + line + "\n")

    def print(self, *args, **kwargs):
        "Prints the given message unless console output is disabled"
        if not self.exists("verbose") or self.get("verbose"):
            print(*args, **kwargs)

    def trace(
        self, echo=False, echo_prefix="", echo_flow=False, log=False, **kwargs
    ) -> Dict[str, Any]:
        """Write a set of key-value pairs to the trace file.

        The pairs are written as a single-line YAML record. Optionally, also
        echo to console and/or write to log file.

        And id and the current time is automatically added using key ``timestamp``.

        Returns the written k/v pairs.
        """
        kwargs["timestamp"] = time.time()
        kwargs["entry_id"] = str(uuid.uuid4())
        line = yaml.dump(kwargs, width=float("inf"), default_flow_style=True).strip()
        if echo or log:
            msg = yaml.dump(kwargs, default_flow_style=echo_flow)
            if log:
                self.log(msg, echo, echo_prefix)
            else:
                for line in msg.splitlines():
                    if echo_prefix:
                        line = echo_prefix + line
                        self.print(line)
        with open(self.tracefile(), "a") as file:
            file.write(line + "\n")
        return kwargs

    # -- FOLDERS AND CHECKPOINTS ----------------------------------------------

    def init_folder(self):
        """Initialize the output folder.

        If the folder does not exists, create it, dump the configuration
        there and return ``True``. Else do nothing and return ``False``.

        """
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            os.makedirs(os.path.join(self.folder, "config"))
            self.save(os.path.join(self.folder, "config.yaml"))
            return True
        return False

    @staticmethod
    def create_from(checkpoint: Dict) -> Config:
        """Create a config from a checkpoint."""
        config = Config()  # round trip to handle deprecated configs
        if "config" in checkpoint and checkpoint["config"] is not None:
            config.load_config(checkpoint["config"].clone())
        if "folder" in checkpoint and checkpoint["folder"] is not None:
            config.folder = checkpoint["folder"]
        return config

    @staticmethod
    def from_options(options: Dict[str, Any] = {}, **more_options) -> Config:
        """Convert given options or kwargs to a Config object.

        Does not perform any checks for correctness."""
        config = Config(load_default=False)
        config.set_all(options, create=True)
        config.set_all(more_options, create=True)
        return config

    def checkpoint_file(self, cpt_id: Union[str, int]) -> str:
        "Return path of checkpoint file for given checkpoint id"
        from kge.misc import is_number

        if is_number(cpt_id, int):
            return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
        else:
            return os.path.join(self.folder, "checkpoint_{}.pt".format(cpt_id))

    def last_checkpoint_number(self) -> Optional[int]:
        "Return number (epoch) of latest checkpoint"
        # stupid implementation, but works
        tried_epoch = 0
        found_epoch = 0
        while tried_epoch < found_epoch + 500:
            tried_epoch += 1
            if os.path.exists(self.checkpoint_file(tried_epoch)):
                found_epoch = tried_epoch
        if found_epoch > 0:
            return found_epoch
        else:
            return None

    @staticmethod
    def best_or_last_checkpoint_file(path: str) -> str:
        """Return best (if present) or last checkpoint path for a given folder path."""
        config = Config(folder=path, load_default=False)
        checkpoint_file = config.checkpoint_file("best")
        if os.path.isfile(checkpoint_file):
            return checkpoint_file
        cpt_epoch = config.last_checkpoint_number()
        if cpt_epoch:
            return config.checkpoint_file(cpt_epoch)
        else:
            raise Exception("Could not find checkpoint in {}".format(path))

    # -- CONVENIENCE METHODS --------------------------------------------------

    def _check(self, key: str, value, allowed_values) -> Any:
        if value not in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}".format(
                    value, key, allowed_values
                )
            )
        return value

    def check(self, key: str, allowed_values) -> Any:
        """Raise an error if value of key is not in allowed.

        If fine, returns value.
        """
        return self._check(key, self.get(key), allowed_values)

    def check_default(self, key: str, allowed_values) -> Any:
        """Raise an error if value or default value of key is not in allowed.

        If fine, returns value.
        """
        return self._check(key, self.get_default(key), allowed_values)

    def check_range(
        self, key: str, min_value, max_value, min_inclusive=True, max_inclusive=True
    ) -> Any:
        value = self.get(key)
        if (
            value < min_value
            or (value == min_value and not min_inclusive)
            or value > max_value
            or (value == max_value and not max_inclusive)
        ):
            raise ValueError(
                "Illegal value {} for key {}; must be in range {}{},{}{}".format(
                    value,
                    key,
                    "[" if min_inclusive else "(",
                    min_value,
                    max_value,
                    "]" if max_inclusive else ")",
                )
            )
        return value

    def logfile(self) -> str:
        folder = self.log_folder if self.log_folder else self.folder
        if folder:
            return os.path.join(folder, "kge.log")
        else:
            return os.devnull

    def tracefile(self) -> str:
        folder = self.log_folder if self.log_folder else self.folder
        if folder:
            return os.path.join(folder, "trace.yaml")
        else:
            return os.devnull


class Configurable:
    """Mix-in class for adding configurations to objects.

    Each configured object has access to a `config` and a `configuration_key` that
    indicates where the object's options can be found in `config`.

    """

    def __init__(self, config: Config, configuration_key: str = None):
        self._init_configuration(config, configuration_key)

    def has_option(self, name: str) -> bool:
        try:
            self.get_option(name)
            return True
        except KeyError:
            return False

    def get_option(self, name: str) -> Any:
        if self.configuration_key:
            return self.config.get_default(self.configuration_key + "." + name)
        else:
            self.config.get_default(name)

    def check_option(self, name: str, allowed_values) -> Any:
        if self.configuration_key:
            return self.config.check_default(
                self.configuration_key + "." + name, allowed_values
            )
        else:
            return self.config.check_default(name, allowed_values)

    def set_option(
        self, name: str, value, create=False, overwrite=Config.Overwrite.Yes, log=False
    ) -> Any:
        if self.configuration_key:
            return self.config.set(
                self.configuration_key + "." + name,
                value,
                create=create,
                overwrite=overwrite,
                log=log,
            )
        else:
            return self.config.set(
                name, value, create=create, overwrite=overwrite, log=log
            )

    def _init_configuration(self, config: Config, configuration_key: Optional[str]):
        r"""Initializes `self.config` and `self.configuration_key`.

        Only after this method has been called, `get_option`, `check_option`, and
        `set_option` should be used. This method is automatically called in the
        constructor of this class, but can also be called by subclasses before calling
        the superclass constructor to allow access to these three methods. May also be
        overridden by subclasses to perform additional configuration.

        """
        self.config = config
        self.configuration_key = configuration_key


def _process_deprecated_options(options: Dict[str, Any]):
    import re

    # renames given key (but not subkeys!)
    def rename_key(old_key, new_key):
        if old_key in options:
            print(
                "Warning: key {} is deprecated; use key {} instead".format(
                    old_key, new_key
                ),
                file=sys.stderr,
            )
            if new_key in options:
                raise ValueError(
                    "keys {} and {} must not both be set".format(old_key, new_key)
                )
            value = options[old_key]
            del options[old_key]
            options[new_key] = value
            return True
        return False

    # renames a value
    def rename_value(key, old_value, new_value):
        if key in options and options.get(key) == old_value:
            print(
                "Warning: value {}={} is deprecated; use value {} instead".format(
                    key, old_value, new_value if new_value != "" else "''"
                ),
                file=sys.stderr,
            )
            options[key] = new_value
            return True
        return False

    # deletes a key, raises error if it does not have the specified value
    def delete_key_with_value(key, value):
        if key in options:
            if options[key] == value:
                print(
                    f"Warning: key {key} is deprecated and has been removed."
                    " Ignoring key since it has default value.",
                    file=sys.stderr,
                )
                del options[key]
            else:
                raise ValueError(f"key {key} is deprecated and has been removed.")

    # renames a set of keys matching a regular expression
    def rename_keys_re(key_regex, replacement):
        renamed_keys = set()
        regex = re.compile(key_regex)
        for old_key in list(options.keys()):
            new_key = regex.sub(replacement, old_key)
            if old_key != new_key:
                rename_key(old_key, new_key)
                renamed_keys.add(new_key)
        return renamed_keys

    # renames a value of keys matching a regular expression
    def rename_value_re(key_regex, old_value, new_value):
        renamed_keys = set()
        regex = re.compile(key_regex)
        for key in options.keys():
            if regex.match(key):
                if rename_value(key, old_value, new_value):
                    renamed_keys.add(key)
        return renamed_keys

    # 10.6.2020
    rename_key("eval.filter_splits", "entity_ranking.filter_splits")
    rename_key("eval.filter_with_test", "entity_ranking.filter_with_test")
    rename_key("eval.tie_handling", "entity_ranking.tie_handling")
    rename_key("eval.hits_at_k_s", "entity_ranking.hits_at_k_s")
    rename_key("eval.chunk_size", "entity_ranking.chunk_size")
    rename_keys_re("^eval\.metrics_per\.", "entity_ranking.metrics_per.")

    # 26.5.2020
    delete_key_with_value("ax_search.fixed_parameters", [])

    # 18.03.2020
    rename_value("train.lr_scheduler", "ConstantLRScheduler", "")

    # 16.03.2020
    rename_key("eval.data", "eval.split")
    rename_key("valid.filter_with_test", "entity_ranking.filter_with_test")

    # 26.02.2020
    rename_value("negative_sampling.implementation", "spo", "triple")
    rename_value("negative_sampling.implementation", "sp_po", "batch")

    # 31.01.2020
    rename_key("negative_sampling.num_samples_s", "negative_sampling.num_samples.s")
    rename_key("negative_sampling.num_samples_p", "negative_sampling.num_samples.p")
    rename_key("negative_sampling.num_samples_o", "negative_sampling.num_samples.o")

    # 10.01.2020
    rename_key("negative_sampling.filter_positives_s", "negative_sampling.filtering.s")
    rename_key("negative_sampling.filter_positives_p", "negative_sampling.filtering.p")
    rename_key("negative_sampling.filter_positives_o", "negative_sampling.filtering.o")

    # 20.12.2019
    for split in ["train", "valid", "test"]:
        old_key = f"dataset.{split}"
        if old_key in options:
            rename_key(old_key, f"dataset.files.{split}.filename")
            options[f"dataset.files.{split}.type"] = "triples"
    for obj in ["entity", "relation"]:
        old_key = f"dataset.{obj}_map"
        if old_key in options:
            rename_key(old_key, f"dataset.files.{obj}_ids.filename")
            options[f"dataset.files.{obj}_ids.type"] = "map"

    # 14.12.2019
    rename_key("negative_sampling.filter_true_s", "negative_sampling.filtering.s")
    rename_key("negative_sampling.filter_true_p", "negative_sampling.filtering.p")
    rename_key("negative_sampling.filter_true_o", "negative_sampling.filtering.o")

    # 14.12.2019
    rename_key("negative_sampling.num_negatives_s", "negative_sampling.num_samples.s")
    rename_key("negative_sampling.num_negatives_p", "negative_sampling.num_samples.p")
    rename_key("negative_sampling.num_negatives_o", "negative_sampling.num_samples.o")

    # 30.10.2019
    rename_value("train.loss", "ce", "kl")
    rename_keys_re(r"\.regularize_args\.weight$", ".regularize_weight")
    for p in [1, 2, 3]:
        for key in rename_value_re(r".*\.regularize$", f"l{p}", "lp"):
            new_key = re.sub(r"\.regularize$", ".regularize_args.p", key)
            options[new_key] = p
            print(f"Set {new_key}={p}.", file=sys.stderr)

    # 21.10.2019
    rename_key("negative_sampling.score_func_type", "negative_sampling.implementation")

    # 1.10.2019
    rename_value("train.type", "1toN", "KvsAll")
    rename_value("train.type", "spo", "1vsAll")
    rename_keys_re(r"^1toN\.", "KvsAll.")
    rename_key("checkpoint.every", "train.checkpoint.every")
    rename_key("checkpoint.keep", "train.checkpoint.keep")
    rename_value("model", "inverse_relations_model", "reciprocal_relations_model")
    rename_keys_re(r"^inverse_relations_model\.", "reciprocal_relations_model.")

    # 30.9.2019
    rename_key(
        "eval.metrics_per_relation_type", "entity_ranking.metrics_per.relation_type"
    )
    rename_key(
        "eval.metrics_per_head_and_tail", "entity_ranking.metrics_per.head_and_tail"
    )
    rename_key(
        "eval.metric_per_argument_frequency_perc",
        "entity_ranking.metrics_per.argument_frequency",
    )
    return options
