import yaml
import pandas as pd
import re
import os
import torch
import sys
import csv
import configparser

from kge.util.misc import kge_base_dir


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
        if ".pt" in args.source.split("/")[-1]:
            checkpoint = args.source
            folder_path = args.source.split("/")[-1]
        else:
            checkpoint = cls.get_checkpoint_from_path(args.source)
            folder_path = args.source
        what = args.what
        if what == "trace":
            trace = folder_path + "/" + "trace.yaml"
            if not os.path.isfile(trace):
                print("Nothing dumped. No trace found at {}".format(trace))
        else:
            raise NotImplementedError

        keymap = {}
        if args.keys:

            suffix = ""
            if not args.csv:
                suffix = "_name"
            with open(args.keys, "r") as keyfile:
                for line in keyfile:
                    keymap[line.rstrip("\n").split("=")[0].strip()+suffix] = line.rstrip("\n").split("=")[1].strip()

        checkpoint = torch.load(f=checkpoint, map_location="cpu")
        config = checkpoint["config"]
        job_id = checkpoint["job_id"]
        epoch = checkpoint["epoch"]

        #TODO if args.csv a predefined set of keymap is used
        # until this is discussed, don't filter out anything
        if args.csv:
            args.eval = True
            args.test = True
            args.train = True
        disjunctions = []
        conjunctions = []
        for arg, pattern in zip(
                                 [args.eval, args.test],
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
        write = None
        if args.csv:
            csvwriter = csv.writer(sys.stdout)
            valid_metric = config.get("valid.metric")
            use_from_trace = [
                "epoch",
                valid_metric,
                "avg_loss",
                "job"
            ]
            use_from_config = [
                "model",
                "dataset.name",
                "train.optimizer",
                "train.optimizer_args.lr"
            ]
            csvwriter.writerow(
                [keymap[key] if key in keymap.keys() else key for key in use_from_config] +
                [keymap[key] if key in keymap.keys() else key for key in use_from_trace]
            )
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
        # unique sequence of job id's that led to the checkpoint
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
        for entry in entries:
            if not (
                     (
                        (entry.get("job") == "train" and entry.get("job_id") in job_ids) or
                        # test and eval entries from "kge test", "kge eval"
                        (entry.get("job") == "eval" and entry.get("resumed_from_job_id") in job_ids) or
                        # eval entries from a training job
                        (entry.get("job") == "eval" and entry.get("parent_job_id") in job_ids)
                     ) and
                     (entry.get("epoch") <= epoch)
            ):
                continue
            if not args.train and entry.get("job") == "train":
                continue
            if args.csv:
                row_config = [config.get(el) for el in use_from_config]
                row_trace = [entry.get(el) for el in use_from_trace]
                csvwriter.writerow(row_config + row_trace)
            else:
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