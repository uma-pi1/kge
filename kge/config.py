from enum import Enum
import collections
import copy
import datetime
import os
import time
import yaml
import uuid
import kge
from kge.util.misc import filename_in_module, is_number


class Config:
    """Configuration options.

    All available options, their types, and their descriptions are defined in
    :file:`config_default.yaml`.
    """

    def __init__(self, folder=None, load_default=True):
        """Initialize with the default configuration"""
        if load_default:
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

    Overwrite = Enum("Overwrite", "Yes No Error")

    def set(self, key, value, create=False, overwrite=Overwrite.Yes):
        """Set value of specified key.

        Nested dictionary values can be accessed via "." (e.g., "job.type").

        If ``create`` is ``False``, raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.

        """
        splits = key.split(".")
        data = self.options

        # flatten path
        path = []
        for i in range(len(splits) - 1):
            create = create or "+++" in data[splits[i]]
            if create and splits[i] not in data:
                data[splits[i]] = dict()
            path.append(splits[i])
            data = data[splits[i]]

        # check correctness of value
        current_value = data.get(splits[-1])
        if current_value is None:
            if not create:
                raise ValueError("key {} not present".format(key))

            if isinstance(value, str) and is_number(value, float):
                value = float(value)
            elif isinstance(value, str) and is_number(value, int):
                value = int(value)
        else:
            if type(value) != type(current_value):
                raise ValueError("key {} has incorrect type".format(key))
            if overwrite == Config.Overwrite.No:
                return current_value
            if overwrite == Config.Overwrite.Error and value != current_value:
                raise ValueError("key {} cannot be overwritten".format(key))

        # all fine, set value
        data[splits[-1]] = value
        return value

    def _import(self, module_name):
        """Imports the specified module configuration.

        Adds the configuration options from kge/model/<module_name>.yaml to
        the configuration. Retains existing module configurations, but verifies
        that fields and their types are correct.

        """

        # load the module_name
        module_config = Config(load_default=False)
        module_config.load(
            filename_in_module(kge.model, "{}.yaml".format(module_name)), create=True
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

    def load(self, filename, create=False, overwrite=Overwrite.Yes):
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
            self._import(new_options.get("model"))
        if "import" in new_options:
            imports = new_options.get("import")
            if not isinstance(imports, list):
                imports = [imports]
            for module_name in imports:
                self._import(module_name)
            del new_options["import"]

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
        line = yaml.dump(
            kwargs, width=float("inf"), default_flow_style=True
        ).strip()
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

    def checkpoint_file(self, epoch):
        "Return path of checkpoint file for given epoch"
        return os.path.join(self.folder, "checkpoint_{:05d}.pt".format(epoch))

    def last_checkpoint(self):
        "Return epoch number of latest checkpoint"
        # stupid implementation, but works
        tried_epoch = 0
        found_epoch = 0
        while tried_epoch < found_epoch + 100:
            tried_epoch += 1
            if os.path.exists(self.checkpoint_file(tried_epoch)):
                found_epoch = tried_epoch
        if found_epoch > 0:
            return found_epoch
        else:
            return None

    # -- CONVENIENCE METHODS --------------------------------------------------

    def check(self, key, allowed_values):
        """Raise an error if value of key is not in allowed"""
        if not self.get(key) in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}".format(
                    self.get(key), key, allowed_values
                )
            )

    def logfile(self):
        return os.path.join(self.folder, "kge.log")

    def tracefile(self):
        return os.path.join(self.folder, "trace.yaml")
