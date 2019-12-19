import time
import os
from collections import OrderedDict
import sys
import torch
import csv
import yaml
import re
import socket

from kge.job import Trace
from kge import Config


## EXPORTED METHODS #####################################################################


def add_dump_parsers(subparsers):
    # 'kge dump' can have associated sub-commands which can have different args
    parser_dump = subparsers.add_parser("dump", help="Dump objects to stdout",)
    subparsers_dump = parser_dump.add_subparsers(
        title="dump_command", dest="dump_command"
    )
    subparsers_dump.required = True
    _add_dump_trace_parser(subparsers_dump)
    _add_dump_checkpoint_parser(subparsers_dump)


def dump(args):
    """Executes the 'kge dump' commands. """
    if args.dump_command == "trace":
        _dump_trace(args)
    elif args.dump_command == "checkpoint":
        _dump_checkpoint(args)
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
        "checkpoint", help=("Dump information stored in a checkpoint"),
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
    """Executes the 'dump checkpoint' command."""

    # Determine checkpoint to use
    if os.path.isfile(args.source):
        checkpoint_file = args.source
    else:
        checkpoint_file = Config.get_best_or_last_checkpoint(args.source)

    # Load the checkpoint and strip some fieleds
    checkpoint = torch.load(checkpoint_file)

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
            "Process and dump trace to stdout and/or csv. The trace will be processed "
            "backwards, starting with a specified job_id."
        ),
    )

    parser_dump_trace.add_argument(
        "source",
        help="A path to either a checkpoint or a job folder.",
        nargs="?",
        default=".",
    )

    parser_dump_trace.add_argument(
        "--checkpoint",
        default=False,
        action="store_const",
        const=True,
        help=(
            "If source is a path to a job folder and --checkpoint is set the best "
            "(if present) or last checkpoint will be used to determine the job_id"
        ),
    )

    parser_dump_trace.add_argument(
        "--job_id",
        default=False,
        help=(
            "Specifies the training job id in the trace "
            "from where to start processing backward"
        ),
    )

    parser_dump_trace.add_argument(
        "--max_epoch",
        default=False,
        help=(
            "Specifies the epoch in the trace"
            "from where to start processing backwards"
        ),
    )

    parser_dump_trace.add_argument(
        "--truncate",
        default=False,
        action="store_const",
        const=True,
        help=(
            "If a checkpoint is used (by providing one explicitly as source or by "
            "using --checkpoint), --truncate will define the max_epoch to process as"
            "specified by the checkpoint"
        ),
    )

    for argument in [
        "--train",
        "--valid",
        "--test",
        "--search",
        "--yaml",
        "--batch",
        "--example",
        "--timeit",
        "--no-header",
    ]:
        parser_dump_trace.add_argument(
            argument, action="store_const", const=True, default=False,
        )
    parser_dump_trace.add_argument(
        "--no-default-keys", "-K", action="store_const", const=True, default=False,
    )

    parser_dump_trace.add_argument("--keysfile", default=False)
    parser_dump_trace.add_argument("--keys", "-k", nargs="*", type=str)


def _dump_trace(args):
    """ Executes the 'dump trace' command."""
    start = time.time()
    if (args.train or args.valid or args.test) and args.search:
        print(
            "--search and --train, --valid, --test are mutually exclusive",
            file=sys.stderr,
        )
        exit(1)
    entry_type_specified = True
    if not (args.train or args.valid or args.test or args.search):
        entry_type_specified = False
        args.train = True
        args.valid = True
        args.test = True

    checkpoint_path = None
    if ".pt" in os.path.split(args.source)[-1]:
        checkpoint_path = args.source
        folder_path = os.path.split(args.source)[0]
    else:
        # determine job_id and epoch from last/best checkpoint automatically
        if args.checkpoint:
            checkpoint_path = Config.get_best_or_last_checkpoint(args.source)
        folder_path = args.source
        if not args.checkpoint and args.truncate:
            raise ValueError(
                "You can only use --truncate when a checkpoint is specified."
                "Consider using --checkpoint or provide a checkpoint file as source"
            )
    trace = os.path.join(folder_path, "trace.yaml")
    if not os.path.isfile(trace):
        sys.stderr.write("No trace found at {}\n".format(trace))
        exit(1)

    keymap = OrderedDict()
    additional_keys = []
    if args.keysfile:
        with open(args.keysfile, "r") as keyfile:
            additional_keys = keyfile.readlines()
    if args.keys:
        additional_keys += args.keys
    for line in additional_keys:
        line.rstrip("\n")
        name_key = line.split("=")
        if len(name_key) == 1:
            name_key += name_key
        keymap[name_key[0]] = name_key[1]

    job_id = None
    epoch = int(args.max_epoch)
    # use job_id and epoch from checkpoint
    if checkpoint_path and args.truncate:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
        epoch = checkpoint["epoch"]
    # only use job_id from checkpoint
    elif checkpoint_path:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
    # override job_id and epoch with user arguments
    if args.job_id:
        job_id = args.job_id
    if not epoch:
        epoch = float("inf")

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
            epoch_of_last=epoch,
        )
    if not entries and (args.search or not entry_type_specified):
        entries = Trace.grep_entries(tracefile=trace, conjunctions=[f"scope: train"],)
        epoch = None
        if entries:
            args.search = True
    if not entries:
        print("No relevant trace entries found.", file=sys.stderr)
        exit(1)

    middle = time.time()
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

        if not args.no_header:
            csv_writer.writerow(
                list(default_attributes.keys()) + [key for key in keymap.keys()]
            )
    # store configs for job_id's s.t. they need to be loaded only once
    configs = {}
    warning_shown = False
    for entry in entries:
        if epoch and not entry.get("epoch") <= float(epoch):
            continue
        # filter out not needed entries from a previous job when
        # a job was resumed from the middle
        if entry.get("job") == "train":
            job_id = entry.get("job_id")
            if entry.get("epoch") > job_epochs[job_id]:
                continue

        # find relevant config file
        child_job_id = entry.get("child_job_id") if "child_job_id" in entry else None
        config_key = (
            entry.get("folder") + "/" + str(child_job_id)
            if args.search
            else entry.get("job_id")
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
                config = get_config_for_job_id(entry.get("job_id"), folder_path)
            configs[config_key] = config

        new_attributes = OrderedDict()
        if config.get_default("model") == "reciprocal_relations_model":
            model = config.get_default("reciprocal_relations_model.base_model.type")
            # the string that substitutes $base_model in keymap if it exists
            subs_model = "reciprocal_relations_model.base_model"
            reciprocal = 1
        else:
            model = config.get_default("model")
            subs_model = model
            reciprocal = 0
        for new_key in keymap.keys():
            lookup = keymap[new_key]
            if "$base_model" in lookup:
                lookup = lookup.replace("$base_model", subs_model)
            try:
                if lookup == "$folder":
                    val = os.path.abspath(folder_path)
                elif lookup == "$checkpoint":
                    val = os.path.abspath(checkpoint_path)
                elif lookup == "$machine":
                    val = socket.gethostname()
                else:
                    val = config.get_default(lookup)
            except:
                # creates empty field if key is not existing
                val = entry.get(lookup)
            if type(val) == bool and val:
                val = 1
            elif type(val) == bool and not val:
                val = 0
            new_attributes[new_key] = val
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
            if entry.get("job") == "train":
                actual_default["split"] = "train"
                actual_default["job"] = "train"
            elif entry.get("job") == "eval":
                actual_default["split"] = entry.get("data")  # test or valid
                if entry.get("resumed_from_job_id"):
                    actual_default["job"] = "eval"  # from "kge eval"
                else:
                    actual_default["job"] = "valid"  # child of training job
            else:
                actual_default["job"] = entry.get("job")
                actual_default["split"] = entry.get("data")
            actual_default["job_id"] = entry.get("job_id").split("-")[0]
            actual_default["model"] = model
            actual_default["reciprocal"] = reciprocal
            # lookup name is in config value is in trace
            actual_default["metric"] = entry.get(config.get_default("valid.metric"))
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
            sys.stdout.write(re.sub("[{}']", "", str(entry)))
            sys.stdout.write("\n")
    end = time.time()
    if args.timeit:
        sys.stdout.write("Grep + processing took {} \n".format(middle - start))
        sys.stdout.write("Writing took {}".format(end - middle))
