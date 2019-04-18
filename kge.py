#!/usr/bin/env python
import datetime
import argparse
import os
import yaml

import kge
from kge import Dataset
from kge import Config
from kge.job import Job
from kge.util.misc import get_git_revision_short_hash, filename_in_module, kge_base_dir


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # default config
    config = Config()

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

    # create main parsers and subparsers
    parser = argparse.ArgumentParser("kge")
    subparsers = parser.add_subparsers(title="command", dest="command")
    subparsers.required = True
    parser_create = subparsers.add_parser(
        "create", help="Create a new job", parents=[parser_conf]
    )
    parser_create.add_argument("config", type=str, nargs="?")
    parser_create.add_argument("--folder", "-f", type=str, help="Output folder to use")
    parser_create.add_argument(
        "--run",
        default=True,
        type=argparse_bool_type,
        help="Whether to immediately run the created job",
    )
    parser_resume = subparsers.add_parser(
        "resume", help="Resume a prior job", parents=[parser_conf]
    )
    parser_resume.add_argument("config", type=str)
    args = parser.parse_args()

    # start command
    if args.command == "create":
        # use toy config file if no config given
        if args.config is None:
            args.config = "examples/toy.yaml"
            print("WARNING: No configuration specified; using " + args.config)

        print("Loading configuration {}...".format(args.config))
        config.load(args.config)

    # resume command
    if args.command == "resume":
        if os.path.isdir(args.config) and os.path.isfile(args.config + "/config.yaml"):
            args.config += "/config.yaml"
        print("Resuming from configuration {}...".format(args.config))
        config.load(args.config)
        config.folder = os.path.dirname(args.config)
        if not os.path.exists(config.folder):
            raise ValueError(
                "{} is not a valid config file for resuming".format(args.config)
            )

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in ["command", "config", "run", "folder"]:
            continue
        if value is not None:
            config.set(key, value)

    # initialize output folder
    if args.command == "create":
        if args.folder is None:  # means: set default
            config.folder = os.path.join(
                kge_base_dir(),
                "local/experiments/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-"
                + config.get("dataset.name")
                + "-"
                + config.get("model"),
            )
        else:
            config.folder = args.folder

    if args.command == "create" and not config.init_folder():
        raise ValueError("output folder {} exists already".format(config.folder))
    config.log("Using folder: {}".format(config.folder))

    # log configuration
    config.log("Configuration:")
    config.log(yaml.dump(config.options), prefix="  ")
    config.log("git commit: {}".format(get_git_revision_short_hash()), prefix="  ")

    if not (args.command == "create" and not args.run):
        # load data
        dataset = Dataset.load(config)

        # let's go
        job = Job.create(config, dataset)
        if args.command == "resume":
            job.resume()
        job.run()
