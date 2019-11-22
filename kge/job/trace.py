import yaml
import pandas as pd
import re
import os
import torch
import sys
import csv
from collections import OrderedDict


from kge.util.misc import kge_base_dir
from kge.config import Config


class Trace:
    """Utility class for handling traces."""

    def __init__(self, tracefile=None, regex_filter=None):
        self.entries = []
        if tracefile:
            self.load(tracefile, regex_filter)

    def load(self, tracefile, regex_filter=None):
        if regex_filter:
            matcher = re.compile(regex_filter)
        with open(tracefile, "r") as file:
            self.kv_pairs = []
            for line in file:
                if regex_filter and not matcher.search(line):
                    continue
                entry = yaml.load(line, Loader=yaml.SafeLoader)
                self.entries.append(entry)

    def filter(self, filter_dict={}):
        def predicate(entry):
            for key, value in filter_dict.items():
                entry_value = entry.get(key)
                if not entry_value or value != entry_value:
                    return False
            return True

        return list(filter(predicate, self.entries))

    def to_dataframe(self, filter_dict={}) -> pd.DataFrame:
        filtered_entries = self.filter(filter_dict)
        return pd.DataFrame(filtered_entries)

    @staticmethod
    def get_metric(entry, metric_name):
        """Return the value of the given metric from a trace entry.

        Understands hits@5 or hits@5_filtered."""
        value = entry.get(metric_name)
        if value:
            return value
        pattern = re.compile("^hits(?:@|_at_)([0-9]+)(_filtered)?$")
        match = pattern.match(metric_name)
        if match:
            k = int(match.group(1))
            if match.group(2):
                return entry.get("hits_at_k_filtered")[k - 1]
            else:
                return entry.get("hits_at_k")[k - 1]
        raise ValueError("metric " + metric_name + " not found")


class ObjectDumper:

    @classmethod
    def dump(cls, args):

        if not(args.train or args.valid or args.test):
            args.train = True
            args.valid = True
            args.test = True

        checkpoint = None
        if ".pt" in args.source.split("/")[-1]:
            checkpoint = args.source
            folder_path = "/".join(args.source.split("/")[:-1])
        else:
            # dermine job_id and epoch from last/best checkpoint automatically
            if args.checkpoint:
                checkpoint = cls.get_checkpoint_from_path(args.source)
            folder_path = args.source
            if not args.checkpoint and args.truncate:
                raise ValueError("You can only use --truncate when a checkpoint is specified. Consider using " +
                                 "--checkpoint or provide a checkpoint file as source")
        what = args.what
        if what == "trace":
            trace = folder_path + "/" + "trace.yaml"
            if not os.path.isfile(trace):
                print("Nothing dumped. No trace found at {}".format(trace))
        else:
            raise NotImplementedError

        keymap = OrderedDict()
        if args.keysfile:
            suffix = ""
            if not args.csv:
                suffix = "_name"
            with open(args.keysfile, "r") as keyfile:
                for line in keyfile:
                    keymap[line.rstrip("\n").split("=")[0].strip()+suffix] = line.rstrip("\n").split("=")[1].strip()

        config = None
        epoch = None
        job_id = None
        epoch = args.max_epoch
        # use job_id and epoch from checkpoint
        if checkpoint and args.truncate:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
            config = checkpoint["config"]
            job_id = checkpoint["job_id"]
            epoch = checkpoint["epoch"]
        # only use job_id from checkpoint
        elif checkpoint:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
            config = checkpoint["config"]
            job_id = checkpoint["job_id"]
        # override job_id and epoch with user arguments if given
        if args.job_id:
            job_id = args.job_id
            config = cls.get_config_for_job_id(job_id, folder_path)
        if args.max_epoch:
            epoch = args.max_epoch
        # if no epoch is specified take all epochs
        if not epoch:
            epoch = float("inf")

        # #TODO debatable if --valid --train --test have effect when csv is specified
        # if args.csv:
        #     args.valid = True
        #     args.test = True
        #     args.train = True
        disjunctions = []
        conjunctions = []
        for arg, pattern in zip(
                                 [args.valid, args.test],
                                 ["(?=.*data: valid)(?=.*job: eval)", "(?=.*data: test)(?=.*job: eval)"]
                            ):
            if arg:
                disjunctions.append(pattern)
        if not args.batch:
            conjunctions.append("scope: epoch")
        # train entries have to be kept initially in all cases to be able to reconstruct the chain
        disjunctions.append("job: train")
        # throw away meta entries
        conjunctions.append("epoch: ")
        entries = []
        conjunctions = [re.compile(pattern) for pattern in conjunctions]
        disjunctions = [re.compile(pattern) for pattern in disjunctions]
        with open(trace, "r") as file:
            for line in file:
                if not(
                        any(map(lambda pattern: pattern.search(line), disjunctions)) and
                        all(map(lambda pattern: pattern.search(line), conjunctions))
                ):
                    continue
                entry = yaml.load(line, Loader=yaml.SafeLoader)
                entries.append(entry)

        #if job_id is not specified, obtain the job id from the last training entry of the trace
        idx = len(entries) - 1
        while(not job_id):
            entry = entries[idx]
            if entry.get("job") == "train":
                job_id = entry.get("job_id")
            idx -= 1

        # determine the unique sequence of job id's that led to job_id
        # this is needed to filter out irrelevant entries
        job_ids = [job_id]
        current_job_id = job_id
        resumed_from = None
        for entry in entries[::-1]:
            if entry.get("job") == "train":
                entry_job_id = entry.get("job_id")
                if entry_job_id == current_job_id:
                    resumed_id = entry.get("resumed_from_job_id")
                    if not resumed_id:
                        break
                    elif resumed_id:
                        resumed_from = resumed_id
                elif entry_job_id == resumed_from:
                    current_job_id = entry_job_id
                    job_ids.insert(0, current_job_id)
                    resumed_id = entry.get("resumed_from_job_id")
                    if not resumed_id:
                        break
                    elif resumed_id:
                        resumed_from = resumed_id
        if args.csv:
            if not config:
                config = cls.get_config_for_job_id(job_id, folder_path)
            csvwriter = csv.writer(sys.stdout)
            valid_metric = config.get("valid.metric")

            # keys from the trace
            use_from_trace = [
                "epoch",
                "avg_loss",
                "data",
                valid_metric
            ]
            # keys from config
            use_from_config = [
                "model",
                "dataset.name",
                "train.optimizer",
                "train.optimizer_args.lr",
            ]
            csvwriter.writerow(
                ["split"] + # the split column helps to distinguish train,eval,valid and test entries
                [key for key in use_from_trace] +
                [key for key in use_from_config]
            )
        for entry in entries:
            if not (
                     (
                        (entry.get("job") == "train" and entry.get("job_id") in job_ids) or
                        # test and eval entries from "kge test", "kge eval"
                        (entry.get("job") == "eval" and entry.get("resumed_from_job_id") in job_ids) or
                        # eval entries from a training job
                        (entry.get("job") == "eval" and entry.get("parent_job_id") in job_ids)
                     ) and
                     (entry.get("epoch") <= float(epoch))
            ):
                continue
            if not args.train and entry.get("job") == "train":
                continue
            if args.csv:
                # load the config for the current job as resumed jobs might have different config parameters
                # TODO should be save when you eg test with some other metric but doublecheck
                config = cls.get_config_for_job_id(entry.get("job_id"), folder_path)
                row_config = [config.get(el) for el in use_from_config]
                split = ""
                if entry.get("job") == "train":
                    split = ["train"]
                elif entry.get("job") == "eval" and entry.get("resumed_from_job_id"):
                    if entry.get("data") == "test":
                        split = ["test"]
                    elif entry.get("data") == "valid":
                        split = ["eval"]
                elif entry.get("job") == "eval" and entry.get("parent_job_id"):
                        split = ["valid"]
                row_trace = [entry.get(el) for el in use_from_trace]
                row_config = [config.get(el) for el in use_from_config]
                csvwriter.writerow(
                    split + row_trace + row_config
                )
            else:
                # TODO: you might want to read the job's config here as well
                #  maybe in further modifications of the functionality we want config entries in the trace
                if keymap:
                    entry.update(keymap)
                sys.stdout.write(yaml.dump(entry).replace("\n",", "))
                sys.stdout.write("\n")

    @classmethod
    def get_checkpoint_from_path(cls, path):
        if "checkpoint_best.pt" in os.listdir(path):
            return path + "/" + "checkpoint_best.pt"
        else:
            checkpoints = sorted(list(filter(lambda file: "checkpoint" in file, os.listdir(path))))
            if len(checkpoints) > 0:
                return path + "/" + checkpoints[-1]
            else:
                print("Nothing was dumped. Did not find a checkpoint in {}".format(path))
                exit()

    @classmethod
    def get_config_for_job_id(cls, job_id, folder_path):
        config = Config(load_default=False)
        config_path = folder_path + "/config/" + job_id.split("-")[0] + ".yaml"
        if os.path.isfile(config_path):
            with open(config_path, "r") as file:
                options = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            raise Exception("Could not find config file for job_id")
        config.options = options
        return config