import yaml
import pandas as pd
import re
import os
import torch
import sys
import csv

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

        checkpoint = torch.load(f=checkpoint, map_location="cpu")
        epoch = checkpoint["epoch"]
        config = checkpoint["config"]

        disjunctions = []
        conjunctions = []
        for arg, pattern in zip(
                                 [args.train, args.eval, args.test],
                                 ["job: train", "job: eval", "(?=.*data: test)(?=.*job: eval)"]
                            ):
            if arg:
                disjunctions.append(pattern)
        if not args.batch:
            # TODO: if --batch is set then also scope:example is included like this
            conjunctions.append("scope: epoch")

        # throw away meta entries
        conjunctions.append("epoch: ")

        write = None
        if args.csv:
            csvfile = open(kge_base_dir() + '/local/dump.csv', 'w', newline="")
            csvwriter = csv.writer(csvfile)
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
            csvwriter.writerow(use_from_config + use_from_trace)
        with open(trace, "r") as file:
            for line in file:
                if not(
                        any(map(lambda string: re.compile(string).search(line), disjunctions)) and
                        all(map(lambda string: re.compile(string).search(line), conjunctions))
                ):
                    continue
                entry = yaml.load(line, Loader=yaml.SafeLoader)
                if entry["epoch"] > epoch:
                    break
                line = line.replace("{", "").replace("}", "")
                if args.csv:
                    row_config = [config.get(el) for el in use_from_config]
                    row_trace = [entry.get(el) for el in use_from_trace]
                    csvwriter.writerow(row_config + row_trace)
                else:
                    sys.stdout.write(line)
            if args.csv:
                csvfile.close()

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