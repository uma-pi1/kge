from enum import Enum
import collections
import copy
import datetime
import os
import time
import yaml
import uuid
from kge.util.misc import filename_in_module, is_number


class Config:
    """Configuration options.

    All available options, their types, and their descriptions are defined in
    :file:`config_default.yaml`.
    """

    def __init__(self, folder=None, load_default=True):
        """Initialize with the default configuration"""
        if load_default:
            import kge

            with open(filename_in_module(kge, "config-default.yaml"), "r") as file:
                self.options = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            self.options = {}

        self.folder = folder
        self.log_prefix = None

    # -- ACCESS METHODS ----------------------------------------------------------------

    def get(self, key, remove_plusplusplus=True):
        """Obtain value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type"). Strips all
        '+++' keys unless `remove_plusplusplus` is set to `False`.

        """
        result = self.options
        for name in key.split("."):
            result = result[name]

        if remove_plusplusplus and isinstance(result, collections.Mapping):

            def do_remove_plusplusplus(option):
                if isinstance(option, collections.Mapping):
                    option.pop("+++", None)
                    for values in option.values():
                        do_remove_plusplusplus(values)

            result = copy.deepcopy(result)
            do_remove_plusplusplus(result)

        return result

    def get_default(self, key):
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

    def get_first_present_key(self, *keys, use_get_default=False):
        "Return the first key for which ``get`` or ``get_default`` finds a value."
        for key in keys:
            try:
                self.get_default(key) if use_get_default else self.get(key)
                return key
            except KeyError:
                pass
        raise KeyError("None of the following keys found: ".format(keys))

    def get_first(self, *keys, use_get_default=False):
        "Return value (or default value) of the first valid key present or KeyError."
        if use_get_default:
            return self.get_default(
                self.get_first_present_key(*keys, use_get_default=True)
            )
        else:
            return self.get(self.get_first_present_key(*keys))

    Overwrite = Enum("Overwrite", "Yes No Error")

    def set(self, key, value, create=False, overwrite=Overwrite.Yes, log=False):

        """Set value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type").

        If ``create`` is ``False`` , raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.

        """
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
                    raise KeyError(
                        (
                            "{} cannot be set because creation of "
                            + "{} is not permitted"
                        ).format(key, ".".join(splits[: (i + 1)]))
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
                raise ValueError("key {} not present".format(key))

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
                    "key {} has incorrect type (expected {}, found {})".format(
                        key, type(current_value), type(value)
                    )
                )
            if overwrite == Config.Overwrite.No:
                return current_value
            if overwrite == Config.Overwrite.Error and value != current_value:
                raise ValueError("key {} cannot be overwritten".format(key))

        # all fine, set value
        data[splits[-1]] = value
        if log:
            self.log("Set {}={}".format(key, value))
        return value

    def _import(self, module_name):
        """Imports the specified module configuration.

        Adds the configuration options from kge/model/<module_name>.yaml to
        the configuration. Retains existing module configurations, but verifies
        that fields and their types are correct.

        """
        import kge.model, kge.model.experimental

        # load the module_name
        module_config = Config(load_default=False)
        module_config.load(
            filename_in_module(
                [kge.model, kge.model.experimental], "{}.yaml".format(module_name)
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

    def set_all(self, new_options, create=False, overwrite=Overwrite.Yes):
        for key, value in Config.flatten(new_options).items():
            self.set(key, value, create, overwrite)

    def load(
        self, filename, create=False, overwrite=Overwrite.Yes, allow_deprecated=True
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

    def save(self, filename):
        """Save this configuration to the given file"""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))

    @staticmethod
    def flatten(options):
        """Returns a dictionary of flattened configuration options."""
        result = {}
        Config.__flatten(options, result)
        return result

    @staticmethod
    def __flatten(options, result, prefix=""):
        for key, value in options.items():
            fullkey = key if prefix == "" else prefix + "." + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    def clone(self, subfolder=None):
        """Return a deep copy"""
        new_config = Config(folder=copy.deepcopy(self.folder), load_default=False)
        new_config.options = copy.deepcopy(self.options)
        if subfolder is not None:
            new_config.folder = os.path.join(self.folder, subfolder)
        return new_config

    # -- LOGGING AND TRACING -----------------------------------------------------------

    def log(self, msg, echo=True, prefix=""):
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
                    print(line)
                file.write(str(datetime.datetime.now()) + " " + line + "\n")

    def trace(
        self, echo=False, echo_prefix="", echo_flow=False, log=False, **kwargs
    ) -> dict:
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
                        print(line)
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

    def checkpoint_file(self, cpt_id):
        "Return path of checkpoint file for given checkpoint id"
        if is_number(cpt_id, int):
            return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(int(cpt_id)))
        else:
            return os.path.join(self.folder, "checkpoint_{}.pt".format(cpt_id))

    def last_checkpoint(self):
        "Return epoch number of latest checkpoint"
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

    # -- CONVENIENCE METHODS --------------------------------------------------

    def _check(self, key, value, allowed_values):
        if not value in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}".format(
                    value, key, allowed_values
                )
            )
        return value

    def check(self, key, allowed_values):
        """Raise an error if value of key is not in allowed.

        If fine, returns value.
        """
        return self._check(key, self.get(key), allowed_values)

    def check_default(self, key, allowed_values):
        """Raise an error if value or default value of key is not in allowed.

        If fine, returns value.
        """
        return self._check(key, self.get_default(key), allowed_values)

    def check_range(
        self, key, min_value, max_value, min_inclusive=True, max_inclusive=True
    ):
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

    def logfile(self):
        return os.path.join(self.folder, "kge.log")

    def tracefile(self):
        return os.path.join(self.folder, "trace.yaml")


class Configurable:
    """Mix-in class for adding configurations to objects.

    Each configured object has access to a `config` and a `configuration_key` that
    indicates where the object's options can be found in `config`.

    """

    def __init__(self, config, configuration_key=None):
        self._init_configuration(config, configuration_key)

    def get_option(self, name):
        if self.configuration_key:
            return self.config.get_default(self.configuration_key + "." + name)
        else:
            self.config.get_default(name)

    def check_option(self, name, allowed_values):
        if self.configuration_key:
            return self.config.check_default(
                self.configuration_key + "." + name, allowed_values
            )
        else:
            return self.config.check_default(name, allowed_values)

    def set_option(self, name, value):
        if self.configuration_key:
            self.config.set(self.configuration_key + "." + name, value)
        else:
            self.config.set(name, value)

    def _init_configuration(self, config, configuration_key):
        r"""Initializes `self.config` and `self.configuration_key`.

        Only after this method has been called, `get_option`, `check_option`, and
        `set_option` should be used. This method is automatically called in the
        constructor of this class, but can also be called by subclasses before calling
        the superclass constructor to allow access to these three methods. May also be
        overridden by subclasess to perform additional configuration.

        """
        self.config = config
        self.configuration_key = configuration_key


def _process_deprecated_options(options):
    # renames given key (but not subkeys!)
    def rename_key(old_key, new_key):
        if old_key in options:
            print(
                "Warning: key {} is deprecated; use {} instead".format(old_key, new_key)
            )
            if new_key in options:
                raise ValueError(
                    "keys {} and {} must not both be set".format(old_key, new_key)
                )
            value = options[old_key]
            del options[old_key]
            options[new_key] = value

    # renames a value
    def rename_value(key, old_value, new_value):
        if key in options and options.get(key) == old_value:
            print(
                "Warning: {}={} is deprecated; use {} instead".format(
                    key, old_value, new_value
                )
            )
            options[key] = new_value

    # renames a set of keys matching a regular expression
    def rename_keys_re(key_regex, replacement):
        import re
        regex = re.compile(key_regex)
        for old_key in options.keys():
            new_key = regex.sub(replacement, old_key)
            if old_key != new_key:
                rename_key(old_key, new_key)

    # 1.10.2019
    rename_value("train.type", "1toN", "KvsAll")
    rename_value("train.type", "spo", "1vsAll")
    rename_keys_re(r"^1toN\.", "KvsAll.")
    rename_key("checkpoint.every", "train.checkpoint.every")
    rename_key("checkpoint.keep", "train.checkpoint.keep")

    # 30.9.2019
    rename_key("eval.metrics_per_relation_type", "eval.metrics_per.relation_type")
    rename_key("eval.metrics_per_head_and_tail", "eval.metrics_per.head_and_tail")
    rename_key(
        "eval.metric_per_argument_frequency_perc", "eval.metrics_per.argument_frequency"
    )

    return options
