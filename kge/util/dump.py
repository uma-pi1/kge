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


def add_dump_parsers(subparsers):
    # 'kge dump' can have associated sub-commands which can have different args
    parser_dump = subparsers.add_parser("dump", help="Dump objects to stdout",)
    subparsers_dump = parser_dump.add_subparsers(
        title="dump_command", dest="dump_command"
    )
    subparsers_dump.required = True

    # 'kge dump trace' command and arguments

    parser_dump_trace = subparsers_dump.add_parser(
        "trace",
        help=(
            "Process and dump trace to stdout and/or csv. The trace will be processed "
            "backwards, starting with a specified job_id."
        ),
    )

    parser_dump_trace.add_argument(
        "source", help="A path to either a checkpoint or a job folder."
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
        "--csv",
        "--batch",
        "--example",
        "--timeit",
        "--no-header",
    ]:
        parser_dump_trace.add_argument(
            argument, action="store_const", const=True, default=False,
        )
    parser_dump_trace.add_argument("--keysfile", default=False)


def get_config_for_job_id(job_id, folder_path):
    config = Config(load_default=True)
    config_path = os.path.join(folder_path, "config", job_id.split("-")[0] + ".yaml")
    if os.path.isfile(config_path):
        config.load(config_path, create=True)
    else:
        raise Exception("Could not find config file for {}".format(job_id))
    return config


def dump_trace(args):
    """ Executes the 'kge dump trace' command."""
    start = time.time()
    if not (args.train or args.valid or args.test):
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
        sys.stdout.write("Nothing dumped. No trace found at {}".format(trace))
        exit()

    keymap = OrderedDict()
    if args.keysfile:
        with open(args.keysfile, "r") as keyfile:
            for line in keyfile:
                keymap[line.rstrip("\n").split("=")[0].strip()] = (
                    line.rstrip("\n").split("=")[1].strip()
                )
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
    middle = time.time()
    if args.csv:
        csv_writer = csv.writer(sys.stdout)
        # dict[new_name] = (lookup_name, where)
        # if where=="config"/"trace" it will be looked up automatically
        # if where=="sep" it must be added in in the write loop separately
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
    for entry in entries:
        if not entry.get("epoch") <= float(epoch):
            continue
        # filter out not needed entries from a previous job when
        # a job was resumed from the middle
        if entry.get("job") == "train":
            job_id = entry.get("job_id")
            if entry.get("epoch") > job_epochs[job_id]:
                continue
        current_job_id = entry.get("job_id")
        if current_job_id in configs.keys():
            config = configs[current_job_id]
        else:
            config = get_config_for_job_id(entry.get("job_id"), folder_path)
            configs[current_job_id] = config
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
        if args.csv:
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
            if entry.get("job") == "eval":
                actual_default["split"] = entry.get("data")  # test or valid
                if entry.get("resumed_from_job_id"):
                    actual_default["job"] = "eval"  # from "kge eval"
                else:
                    actual_default["job"] = "valid"  # child of training job
            actual_default["job_id"] = entry.get("job_id").split("-")[0]
            actual_default["model"] = model
            actual_default["reciprocal"] = reciprocal
            # lookup name is in config value is in trace
            actual_default["metric"] = entry.get(config.get_default("valid.metric"))
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


def dump(args):
    """Executes the 'kge dump' commands. """
    if args.dump_command == "trace":
        dump_trace(args)
        exit()
