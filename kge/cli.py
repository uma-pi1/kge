#!/usr/bin/env python
import datetime
import argparse
import os
import sys
import traceback
import yaml

from kge import Dataset
from kge import Config
from kge.job import Job
from kge.misc import get_git_revision_short_hash, kge_base_dir, is_number
from kge.util.dump import add_dump_parsers, dump
from kge.util.io import get_checkpoint_file, load_checkpoint
from kge.util.package import package_model, add_package_parser


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def process_meta_command(args, meta_command, fixed_args):
    """Process&update program arguments for meta commands.

    `meta_command` is the name of a special command, which fixes all key-value arguments
    given in `fixed_args` to the specified value. `fxied_args` should contain key
    `command` (for the actual command being run).

    """
    if args.command == meta_command:
        for k, v in fixed_args.items():
            if k != "command" and vars(args)[k] and vars(args)[k] != v:
                raise ValueError(
                    "invalid argument for '{}' command: --{} {}".format(
                        meta_command, k, v
                    )
                )
            vars(args)[k] = v


def create_parser(config, additional_args=[]):
    # define short option names
    short_options = {
        "dataset.name": "-d",
        "job.type": "-j",
        "train.max_epochs": "-e",
        "model": "-m",
    }

    # create parser for config
    parser_conf = argparse.ArgumentParser(add_help=False)
    for key, value in Config.flatten(config.options).items():
        short = short_options.get(key)
        argtype = type(value)
        if argtype == bool:
            argtype = argparse_bool_type
        if short:
            parser_conf.add_argument("--" + key, short, type=argtype)
        else:
            parser_conf.add_argument("--" + key, type=argtype)

    # add additional arguments
    for key in additional_args:
        parser_conf.add_argument(key)

    # add argument to abort on outdated data
    parser_conf.add_argument(
        "--abort-when-cache-outdated",
        action="store_const",
        const=True,
        default=False,
        help="Abort processing when an outdated cached dataset file is found "
        "(see description of `dataset.pickle` configuration key). "
        "Default is to recompute such cache files.",
    )

    # create main parsers and subparsers
    parser = argparse.ArgumentParser("kge")
    subparsers = parser.add_subparsers(title="command", dest="command")
    subparsers.required = True

    # start and its meta-commands
    parser_start = subparsers.add_parser(
        "start", help="Start a new job (create and run it)", parents=[parser_conf]
    )
    parser_create = subparsers.add_parser(
        "create", help="Create a new job (but do not run it)", parents=[parser_conf]
    )
    for p in [parser_start, parser_create]:
        p.add_argument("config", type=str, nargs="?")
        p.add_argument("--folder", "-f", type=str, help="Output folder to use")
        p.add_argument(
            "--run",
            default=p is parser_start,
            type=argparse_bool_type,
            help="Whether to immediately run the created job",
        )

    # resume and its meta-commands
    parser_resume = subparsers.add_parser(
        "resume", help="Resume a prior job", parents=[parser_conf]
    )
    parser_eval = subparsers.add_parser(
        "eval", help="Evaluate the result of a prior job", parents=[parser_conf]
    )
    parser_valid = subparsers.add_parser(
        "valid",
        help="Evaluate the result of a prior job using validation data",
        parents=[parser_conf],
    )
    parser_test = subparsers.add_parser(
        "test",
        help="Evaluate the result of a prior job using test data",
        parents=[parser_conf],
    )
    for p in [parser_resume, parser_eval, parser_valid, parser_test]:
        p.add_argument("config", type=str)
        p.add_argument(
            "--checkpoint",
            type=str,
            help=(
                "Which checkpoint to use: 'default', 'last', 'best', a number "
                "or a file name"
            ),
            default="default",
        )
    add_dump_parsers(subparsers)
    add_package_parser(subparsers)
    return parser


def main():
    # default config
    config = Config()

    # now parse the arguments
    parser = create_parser(config)
    args, unknown_args = parser.parse_known_args()

    # If there where unknown args, add them to the parser and reparse. The correctness
    # of these arguments will be checked later.
    if len(unknown_args) > 0:
        parser = create_parser(
            config, filter(lambda a: a.startswith("--"), unknown_args)
        )
        args = parser.parse_args()

    # process meta-commands
    process_meta_command(args, "create", {"command": "start", "run": False})
    process_meta_command(args, "eval", {"command": "resume", "job.type": "eval"})
    process_meta_command(
        args, "test", {"command": "resume", "job.type": "eval", "eval.split": "test"}
    )
    process_meta_command(
        args, "valid", {"command": "resume", "job.type": "eval", "eval.split": "valid"}
    )
    # dump command
    if args.command == "dump":
        dump(args)
        exit()

    # package command
    if args.command == "package":
        package_model(args)
        exit()

    # start command
    if args.command == "start":
        # use toy config file if no config given
        if args.config is None:
            args.config = kge_base_dir() + "/" + "examples/toy-complex-train.yaml"
            print(
                "WARNING: No configuration specified; using " + args.config,
                file=sys.stderr,
            )

        if args.verbose != False:
            print("Loading configuration {}...".format(args.config))
        config.load(args.config)

    # resume command
    if args.command == "resume":
        if os.path.isdir(args.config) and os.path.isfile(args.config + "/config.yaml"):
            args.config += "/config.yaml"
        if args.verbose != False:
            print("Resuming from configuration {}...".format(args.config))
        config.load(args.config)
        config.folder = os.path.dirname(args.config)
        if not config.folder:
            config.folder = "."
        if not os.path.exists(config.folder):
            raise ValueError(
                "{} is not a valid config file for resuming".format(args.config)
            )

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in [
            "command",
            "config",
            "run",
            "folder",
            "checkpoint",
            "abort_when_cache_outdated",
        ]:
            continue
        if value is not None:
            if key == "search.device_pool":
                value = "".join(value).split(",")
            try:
                if isinstance(config.get(key), bool):
                    value = argparse_bool_type(value)
            except KeyError:
                pass
            config.set(key, value)
            if key == "model":
                config._import(value)

    # initialize output folder
    if args.command == "start":
        if args.folder is None:  # means: set default
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            config.folder = os.path.join(
                kge_base_dir(),
                "local",
                "experiments",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + config_name,
            )
        else:
            config.folder = args.folder

    # catch errors to log them
    try:
        if args.command == "start" and not config.init_folder():
            raise ValueError("output folder {} exists already".format(config.folder))
        config.log("Using folder: {}".format(config.folder))

        # determine checkpoint to resume (if any)
        if hasattr(args, "checkpoint"):
            checkpoint_file = get_checkpoint_file(config, args.checkpoint)

        # disable processing of outdated cached dataset files globally
        Dataset._abort_when_cache_outdated = args.abort_when_cache_outdated

        # log configuration
        config.log("Configuration:")
        config.log(yaml.dump(config.options), prefix="  ")
        config.log("git commit: {}".format(get_git_revision_short_hash()), prefix="  ")

        # set random seeds
        if config.get("random_seed.python") > -1:
            import random

            random.seed(config.get("random_seed.python"))
        if config.get("random_seed.torch") > -1:
            import torch

            torch.manual_seed(config.get("random_seed.torch"))
        if config.get("random_seed.numpy") > -1:
            import numpy.random

            numpy.random.seed(config.get("random_seed.numpy"))

        # let's go
        if args.command == "start" and not args.run:
            config.log("Job created successfully.")
        else:
            # load data
            dataset = Dataset.create(config)

            # let's go
            if args.command == "resume":
                if checkpoint_file is not None:
                    checkpoint = load_checkpoint(
                        checkpoint_file, config.get("job.device")
                    )
                    job = Job.create_from(
                        checkpoint, new_config=config, dataset=dataset
                    )
                else:
                    job = Job.create(config, dataset)
                    job.config.log(
                        "No checkpoint found or specified, starting from scratch..."
                    )
            else:
                job = Job.create(config, dataset)
            job.run()
    except BaseException as e:
        tb = traceback.format_exc()
        config.log(tb, echo=False)
        raise e from None


if __name__ == "__main__":
    main()
