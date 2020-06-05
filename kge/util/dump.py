import time
import os
from collections import OrderedDict
import sys
import torch
import csv
import yaml
import re
import socket
import copy

from kge.job import Trace
from kge import Config


## EXPORTED METHODS #####################################################################


def add_dump_parsers(subparsers):
    # 'kge dump' can have associated sub-commands which can have different args
    parser_dump = subparsers.add_parser("dump", help="Dump objects to stdout")
    subparsers_dump = parser_dump.add_subparsers(
        title="dump_command", dest="dump_command"
    )
    subparsers_dump.required = True
    _add_dump_trace_parser(subparsers_dump)
    _add_dump_checkpoint_parser(subparsers_dump)
    _add_dump_config_parser(subparsers_dump)


def dump(args):
    """Execute the 'kge dump' commands. """
    if args.dump_command == "trace":
        _dump_trace(args)
    elif args.dump_command == "checkpoint":
        _dump_checkpoint(args)
    elif args.dump_command == "config":
        _dump_config(args)
    else:
        raise ValueError()


def get_config_for_job_id(job_id, folder_path):
    config = Config(load_default=True)
    if job_id:
        config_path = os.path.join(
            folder_path, "config", job_id.split("-")[0] + ".yaml"
        )
    else:
        config_path = os.path.join(folder_path, "config.yaml")
    if os.path.isfile(config_path):
        config.load(config_path, create=True)
    else:
        raise Exception("Could not find config file for {}".format(job_id))
    return config


### DUMP CHECKPOINT #####################################################################


def _add_dump_checkpoint_parser(subparsers_dump):
    parser_dump_checkpoint = subparsers_dump.add_parser(
        "checkpoint", help=("Dump information stored in a checkpoint")
    )
    parser_dump_checkpoint.add_argument(
        "source",
        help="A path to either a checkpoint or a job folder (then uses best or, "
        "if not present, last checkpoint).",
        nargs="?",
        default=".",
    )
    parser_dump_checkpoint.add_argument(
        "--keys",
        "-k",
        type=str,
        nargs="*",
        help="List of keys to include (separated by space)",
    )


def _dump_checkpoint(args):
    """Execute the 'dump checkpoint' command."""

    # Determine checkpoint to use
    if os.path.isfile(args.source):
        checkpoint_file = args.source
    else:
        checkpoint_file = Config.best_or_last_checkpoint_file(args.source)

    # Load the checkpoint and strip some fieleds
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    # Dump it
    print(f"# Dump of checkpoint: {checkpoint_file}")
    excluded_keys = {"model", "optimizer_state_dict"}
    if args.keys is not None:
        excluded_keys = {key for key in excluded_keys if key not in args.keys}
        excluded_keys = excluded_keys.union(
            {key for key in checkpoint if key not in args.keys}
        )
    excluded_keys = {key for key in excluded_keys if key in checkpoint}
    for key in excluded_keys:
        del checkpoint[key]
    if excluded_keys:
        print(f"# Excluded keys: {excluded_keys}")
    yaml.dump(checkpoint, sys.stdout)


### DUMP TRACE ##########################################################################


def _add_dump_trace_parser(subparsers_dump):
    parser_dump_trace = subparsers_dump.add_parser(
        "trace",
        help=(
            "Dump the trace of a job to stdout as CSV (default) or YAML. The tracefile"
            " is processed backwards starting from the last entry. Further options"
            " allow to start processing from a particular checkpoint, job_id, or"
            " epoch number."
        ),
    )
    parser_dump_trace.add_argument(
        "source",
        help="A path to either a checkpoint or a job folder.",
        nargs="?",
        default=".",
    )
    parser_dump_trace.add_argument(
        "--train",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from training jobs (enabled when none of --train, --valid,"
            " or --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--valid",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from validation or evaluation jobs on the valid split"
            " (enabled when none of --train, --valid, or --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--test",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from evaluation on the test data split (enabled when "
            "  none of --train, --valid, or --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--search",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Dump the tracefile of a search job. The best result of every "
            " search trial is dumped. The options --train, --valid, --test,"
            " --truncate, --job_id, --checkpoint, --batch, and --example are not"
            " applicable."
        ),
    )
    parser_dump_trace.add_argument(
        "--keysfile",
        default=False,
        help=(
            "A path to a file which contains lines in the format"
            " 'new_key_name'='key_name'. For every line in the keys file, the command"
            " searches the value of 'key_name' in the trace entries (first) and"
            " config (second) and adds a respective column in the CSV file with name"
            " 'new_key_name'. Additionally, for 'key_name' the special keys '$folder',"
            " '$machine' '$checkpoint' and '$base_model' can be used."
        ),
    )
    parser_dump_trace.add_argument(
        "--keys",
        "-k",
        nargs="*",
        type=str,
        help=(
            "A list of 'key' entries (separated by space). Each 'key' has form"
            " 'new_key_name=key_name' or 'key_name'. This adds a column as in the"
            " --keysfile option. When only 'key_name' is provided, it is also used as"
            " the column name in the CSV file."
        ),
    )
    parser_dump_trace.add_argument(
        "--checkpoint",
        default=False,
        action="store_const",
        const=True,
        help=(
            "If source is a path to a job folder and --checkpoint is set, the best"
            " (if present) or last checkpoint is used to determine the job_id from"
            " where the tracefile is processed backwards."
        ),
    )
    parser_dump_trace.add_argument(
        "--job_id",
        default=False,
        help=(
            "Specifies the training job_id in the tracefile from where to start"
            " processing backwards when no checkpoint is specified. If not provided,"
            " the job_id of the last training job entry in the tracefile is used."
        ),
    )
    parser_dump_trace.add_argument(
        "--truncate",
        action="store",
        default=False,
        const=True,
        nargs="?",
        help=(
            "Takes an integer argument which defines the maximum epoch number from"
            " where the tracefile is processed backwards. If not provided, all epochs"
            " are included (the epoch number can still be bounded by a specified"
            " job_id or checkpoint). When a checkpoint is specified, (by providing one"
            " explicitly as source or by using --checkpoint), --truncate can"
            " additionally be enabled without an argument which sets the maximum epoch"
            " number to the epoch provided by the checkpoint."
        ),
    )
    parser_dump_trace.add_argument(
        "--yaml",
        action="store_const",
        const=True,
        default=False,
        help="Dump YAML instead of CSV.",
    )
    parser_dump_trace.add_argument(
        "--batch",
        action="store_const",
        const=True,
        default=False,
        help="Include entries on batch level.",
    )
    parser_dump_trace.add_argument(
        "--example",
        action="store_const",
        const=True,
        default=False,
        help="Include entries on example level.",
    )
    parser_dump_trace.add_argument(
        "--no-header",
        action="store_const",
        const=True,
        default=False,
        help="Exclude column names (header) from the CSV file.",
    )
    parser_dump_trace.add_argument(
        "--no-default-keys",
        "-K",
        action="store_const",
        const=True,
        default=False,
        help="Exclude default keys from the CSV file.",
    )
    parser_dump_trace.add_argument(
        "--list-keys",
        action="store",
        const=True,
        default=False,
        nargs="?",
        help="Output the CSV default keys and all usable keys for --keysfile and --keys"
        " for the given configuration of options. Takes an optional string argument"
        " which separates the listed keys (default comma), e.g. use $'\\n' to display"
        " every key on a new line.",
    )


def _dump_trace(args):
    """Execute the 'dump trace' command."""
    if (
        args.train
        or args.valid
        or args.test
        or args.truncate
        or args.job_id
        or args.checkpoint
        or args.batch
        or args.example
    ) and args.search:
        sys.exit(
            "--search and any of --train, --valid, --test, --truncate, --job_id,"
            " --checkpoint, --batch, --example are mutually exclusive"
        )

    entry_type_specified = True
    if not (args.train or args.valid or args.test or args.search):
        entry_type_specified = False
        args.train = True
        args.valid = True
        args.test = True

    truncate_flag = False
    truncate_epoch = None
    if isinstance(args.truncate, bool) and args.truncate:
        truncate_flag = True
    elif not isinstance(args.truncate, bool):
        if not args.truncate.isdigit():
            sys.exit("Integer argument or no argument for --truncate must be used")
        truncate_epoch = int(args.truncate)

    checkpoint_path = None
    if ".pt" in os.path.split(args.source)[-1]:
        checkpoint_path = args.source
        folder_path = os.path.split(args.source)[0]
    else:
        # determine job_id and epoch from last/best checkpoint automatically
        if args.checkpoint:
            checkpoint_path = Config.best_or_last_checkpoint_file(args.source)
        folder_path = args.source
    if not checkpoint_path and truncate_flag:
        sys.exit(
            "--truncate can only be used as a flag when a checkpoint is specified."
            " Consider specifying a checkpoint or use an integer argument for the"
            " --truncate option"
        )
    if checkpoint_path and args.job_id:
        sys.exit(
            "--job_id cannot be used together with a checkpoint as the checkpoint"
            " already specifies the job_id"
        )
    trace = os.path.join(folder_path, "trace.yaml")
    if not os.path.isfile(trace):
        sys.exit(f"No file 'trace.yaml' found at {os.path.abspath(folder_path)}")

    # process additional keys from --keys and --keysfile
    keymap = OrderedDict()
    additional_keys = []
    if args.keysfile:
        with open(args.keysfile, "r") as keyfile:
            additional_keys = keyfile.readlines()
    if args.keys:
        additional_keys += args.keys
    for line in additional_keys:
        line = line.rstrip("\n").replace(" ", "")
        name_key = line.split("=")
        if len(name_key) == 1:
            name_key += name_key
        keymap[name_key[0]] = name_key[1]

    job_id = None
    # use job_id and truncate_epoch from checkpoint
    if checkpoint_path and truncate_flag:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
        truncate_epoch = checkpoint["epoch"]
    # only use job_id from checkpoint
    elif checkpoint_path:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
    # no checkpoint specified job_id might have been set manually
    elif args.job_id:
        job_id = args.job_id
    # don't restrict epoch number in case it has not been specified yet
    if not truncate_epoch:
        truncate_epoch = float("inf")

    entries, job_epochs = [], {}
    if not args.search:
        entries, job_epochs = Trace.grep_training_trace_entries(
            tracefile=trace,
            train=args.train,
            test=args.test,
            valid=args.valid,
            example=args.example,
            batch=args.batch,
            job_id=job_id,
            epoch_of_last=truncate_epoch,
        )
    if not entries and (args.search or not entry_type_specified):
        entries = Trace.grep_entries(tracefile=trace, conjunctions=[f"scope: train"])
        truncate_epoch = None
        if entries:
            args.search = True
    if not entries and entry_type_specified:
        sys.exit(
            "No relevant trace entries found. If this was a trace from a search"
            " job, dont use any of --train --valid --test."
        )
    elif not entries:
        sys.exit("No relevant trace entries found.")

    if args.list_keys:
        all_trace_keys = set()

    if not args.yaml:
        csv_writer = csv.writer(sys.stdout)
        # dict[new_name] = (lookup_name, where)
        # if where=="config"/"trace" it will be looked up automatically
        # if where=="sep" it must be added in in the write loop separately
        if args.no_default_keys:
            default_attributes = OrderedDict()
        else:
            default_attributes = OrderedDict(
                [
                    ("job_id", ("job_id", "sep")),
                    ("dataset", ("dataset.name", "config")),
                    ("model", ("model", "sep")),
                    ("reciprocal", ("reciprocal", "sep")),
                    ("job", ("job", "sep")),
                    ("job_type", ("type", "trace")),
                    ("split", ("split", "sep")),
                    ("epoch", ("epoch", "trace")),
                    ("avg_loss", ("avg_loss", "trace")),
                    ("avg_penalty", ("avg_penalty", "trace")),
                    ("avg_cost", ("avg_cost", "trace")),
                    ("metric_name", ("valid.metric", "config")),
                    ("metric", ("metric", "sep")),
                ]
            )
            if args.search:
                default_attributes["child_folder"] = ("folder", "trace")
                default_attributes["child_job_id"] = ("child_job_id", "sep")

        if not (args.no_header or args.list_keys):
            csv_writer.writerow(
                list(default_attributes.keys()) + [key for key in keymap.keys()]
            )
    # store configs for job_id's s.t. they need to be loaded only once
    configs = {}
    warning_shown = False
    for entry in entries:
        current_epoch = entry.get("epoch")
        job_type = entry.get("job")
        job_id = entry.get("job_id")
        if truncate_epoch and not current_epoch <= float(truncate_epoch):
            continue
        # filter out entries not relevant to the unique training sequence determined
        # by the options; not relevant for search
        if job_type == "train":
            if current_epoch > job_epochs[job_id]:
                continue
        elif job_type == "eval":
            if "resumed_from_job_id" in entry:
                if current_epoch > job_epochs[entry.get("resumed_from_job_id")]:
                    continue
            elif "parent_job_id" in entry:
                if current_epoch > job_epochs[entry.get("parent_job_id")]:
                    continue
        # find relevant config file
        child_job_id = entry.get("child_job_id") if "child_job_id" in entry else None
        config_key = (
            entry.get("folder") + "/" + str(child_job_id) if args.search else job_id
        )
        if config_key in configs.keys():
            config = configs[config_key]
        else:
            if args.search:
                if not child_job_id and not warning_shown:
                    # This warning is from Dec 19, 2019. TODO remove
                    print(
                        "Warning: You are dumping the trace of an older search job. "
                        "This is fine only if "
                        "the config.yaml files in each subfolder have not been modified "
                        "after running the corresponding training job.",
                        file=sys.stderr,
                    )
                    warning_shown = True
                config = get_config_for_job_id(
                    child_job_id, os.path.join(folder_path, entry.get("folder"))
                )
                entry["type"] = config.get("train.type")
            else:
                config = get_config_for_job_id(job_id, folder_path)
            configs[config_key] = config
        if args.list_keys:
            all_trace_keys.update(entry.keys())
            continue
        new_attributes = OrderedDict()
        # when training was reciprocal, use the base_model as model
        if config.get_default("model") == "reciprocal_relations_model":
            model = config.get_default("reciprocal_relations_model.base_model.type")
            # the string that substitutes $base_model in keymap if it exists
            subs_model = "reciprocal_relations_model.base_model"
            reciprocal = 1
        else:
            model = config.get_default("model")
            subs_model = model
            reciprocal = 0
        # search for the additional keys from --keys and --keysfile
        for new_key in keymap.keys():
            lookup = keymap[new_key]
            # search for special keys
            value = None
            if lookup == "$folder":
                value = os.path.abspath(folder_path)
            elif lookup == "$checkpoint" and checkpoint_path:
                value = os.path.abspath(checkpoint_path)
            elif lookup == "$machine":
                value = socket.gethostname()
            if "$base_model" in lookup:
                lookup = lookup.replace("$base_model", subs_model)
            # search for ordinary keys; start searching in trace entry then config
            if not value:
                value = entry.get(lookup)
            if not value:
                try:
                    value = config.get_default(lookup)
                except:
                    pass  # value stays None; creates empty field in csv
            if value and isinstance(value, bool):
                value = 1
            elif not value and isinstance(value, bool):
                value = 0
            new_attributes[new_key] = value
        if not args.yaml:
            # find the actual values for the default attributes
            actual_default = default_attributes.copy()
            for new_key in default_attributes.keys():
                lookup, where = default_attributes[new_key]
                if where == "config":
                    actual_default[new_key] = config.get(lookup)
                elif where == "trace":
                    actual_default[new_key] = entry.get(lookup)
            # keys with separate treatment
            # "split" in {train,test,valid} for the datatype
            # "job" in {train,eval,valid,search}
            if job_type == "train":
                if "split" in entry:
                    actual_default["split"] = entry.get("split")
                else:
                    actual_default["split"] = "train"
                actual_default["job"] = "train"
            elif job_type == "eval":
                if "split" in entry:
                    actual_default["split"] = entry.get("split")  # test or valid
                else:
                    # deprecated
                    actual_default["split"] = entry.get("data")  # test or valid
                if entry.get("resumed_from_job_id"):
                    actual_default["job"] = "eval"  # from "kge eval"
                else:
                    actual_default["job"] = "valid"  # child of training job
            else:
                actual_default["job"] = job_type
                if "split" in entry:
                    actual_default["split"] = entry.get("split")
                else:
                    # deprecated
                    actual_default["split"] = entry.get("data")  # test or valid
            actual_default["job_id"] = job_id.split("-")[0]
            actual_default["model"] = model
            actual_default["reciprocal"] = reciprocal
            # lookup name is in config value is in trace
            actual_default["metric"] = entry.get(config.get_default("valid.metric"))
            if args.search:
                actual_default["child_job_id"] = entry.get("child_job_id").split("-")[0]
            for key in list(actual_default.keys()):
                if key not in default_attributes:
                    del actual_default[key]
            csv_writer.writerow(
                [actual_default[new_key] for new_key in actual_default.keys()]
                + [new_attributes[new_key] for new_key in new_attributes.keys()]
            )
        else:
            entry.update({"reciprocal": reciprocal, "model": model})
            if keymap:
                entry.update(new_attributes)
            print(entry)
    if args.list_keys:
        # only one config needed
        config = configs[list(configs.keys())[0]]
        options = Config.flatten(config.options)
        options = sorted(
            filter(lambda opt: "+++" not in opt, options), key=lambda opt: opt.lower()
        )
        if isinstance(args.list_keys, bool):
            sep = ", "
        else:
            sep = args.list_keys
        print("Default keys for CSV: ")
        print(*default_attributes.keys(), sep=sep)
        print("")
        print("Special keys: ")
        print(*["$folder", "$checkpoint", "$machine", "$base_model"], sep=sep)
        print("")
        print("Keys found in trace: ")
        print(*sorted(all_trace_keys), sep=sep)
        print("")
        print("Keys found in config: ")
        print(*options, sep=sep)


### DUMP CONFIG ########################################################################


def _add_dump_config_parser(subparsers_dump):
    parser_dump_config = subparsers_dump.add_parser(
        "config", help=("Dump a configuration")
    )
    parser_dump_config.add_argument(
        "source",
        help="A path to either a checkpoint, a config file, or a job folder.",
        nargs="?",
        default=".",
    )

    parser_dump_config.add_argument(
        "--minimal",
        "-m",
        default=False,
        action="store_const",
        const=True,
        help="Only dump configuration options different from the default configuration (default)",
    )
    parser_dump_config.add_argument(
        "--raw",
        "-r",
        default=False,
        action="store_const",
        const=True,
        help="Dump the config as is",
    )
    parser_dump_config.add_argument(
        "--full",
        "-f",
        default=False,
        action="store_const",
        const=True,
        help="Add all values from the default configuration before dumping the config",
    )

    parser_dump_config.add_argument(
        "--include",
        "-i",
        type=str,
        nargs="*",
        help="List of keys to include (separated by space). "
        "All subkeys are also included. Cannot be used with --raw.",
    )

    parser_dump_config.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="*",
        help="List of keys to exclude (separated by space). "
        "All subkeys are also exluded. Applied after --include. "
        "Cannot be used with --raw.",
    )


def _dump_config(args):
    """Execute the 'dump config' command."""
    if not (args.raw or args.full or args.minimal):
        args.minimal = True

    if args.raw + args.full + args.minimal != 1:
        raise ValueError("Exactly one of --raw, --full, or --minimal must be set")

    if args.raw and (args.include or args.exclude):
        raise ValueError(
            "--include and --exclude cannot be used with --raw "
            "(use --full or --minimal instead)."
        )

    config = Config()
    config_file = None
    if os.path.isdir(args.source):
        config_file = os.path.join(args.source, "config.yaml")
        config.load(config_file)
    elif ".yaml" in os.path.split(args.source)[-1]:
        config_file = args.source
        config.load(config_file)
    else:  # a checkpoint
        checkpoint = torch.load(args.source, map_location="cpu")
        if args.raw:
            config = checkpoint["config"]
        else:
            config.load_config(checkpoint["config"])

    def print_options(options):
        # drop all arguments that are not included
        if args.include:
            args.include = set(args.include)
            options_copy = copy.deepcopy(options)
            for key in options_copy.keys():
                prefix = key
                keep = False
                while True:
                    if prefix in args.include:
                        keep = True
                        break
                    else:
                        last_dot_index = prefix.rfind(".")
                        if last_dot_index < 0:
                            break
                        else:
                            prefix = prefix[:last_dot_index]
                if not keep:
                    del options[key]

        # remove all arguments that are excluded
        if args.exclude:
            args.exclude = set(args.exclude)
            options_copy = copy.deepcopy(options)
            for key in options_copy.keys():
                prefix = key
                while True:
                    if prefix in args.exclude:
                        del options[key]
                        break
                    else:
                        last_dot_index = prefix.rfind(".")
                        if last_dot_index < 0:
                            break
                        else:
                            prefix = prefix[:last_dot_index]

        # convert the remaining options to a Config and print it
        config = Config(load_default=False)
        config.set_all(options, create=True)
        print(yaml.dump(config.options))

    if args.raw:
        if config_file:
            with open(config_file, "r") as f:
                print(f.read())
        else:
            print_options(config.options)
    elif args.full:
        print_options(config.options)
    else:  # minimal
        default_config = Config()
        imports = config.get("import")
        if imports is not None:
            if not isinstance(imports, list):
                imports = [imports]
            for module_name in imports:
                default_config._import(module_name)
        default_options = Config.flatten(default_config.options)
        new_options = Config.flatten(config.options)
        minimal_options = {}

        for option, value in new_options.items():
            if option not in default_options or default_options[option] != value:
                minimal_options[option] = value

        # always retain all imports
        if imports is not None:
            minimal_options["import"] = list(set(imports))

        print_options(minimal_options)
