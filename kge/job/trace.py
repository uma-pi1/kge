import yaml
import pandas as pd
import re
import os
import torch
import sys
import csv
from collections import OrderedDict
import subprocess
import time


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
    def dump_trace(cls, args):
        start = time.time()
        if not(args.train or args.valid or args.test):
            args.train = True
            args.valid = True
            args.test = True

        checkpoint = None
        if ".pt" in args.source.split("/")[-1]:
            checkpoint = args.source
            folder_path = "/".join(args.source.split("/")[:-1])
        else:
            # determine job_id and epoch from last/best checkpoint automatically
            if args.checkpoint:
                checkpoint = cls.get_checkpoint_from_path(args.source)
            folder_path = args.source
            if not args.checkpoint and args.truncate:
                raise ValueError(
                    "You can only use --truncate when a checkpoint is specified."  
                    "Consider  using --checkpoint or provide a checkpoint file as source"
                )
        trace = folder_path + "/" + "trace.yaml"
        if not os.path.isfile(trace):
            sys.stdout.write("Nothing dumped. No trace found at {}".format(trace))
            exit()

        keymap = OrderedDict()
        if args.keysfile:
            with open(args.keysfile, "r") as keyfile:
                for line in keyfile:
                    keymap[line.rstrip("\n").split("=")[0].strip()] \
                    = line.rstrip("\n").split("=")[1].strip()
        job_id = None
        epoch = args.max_epoch
        # use job_id and epoch from checkpoint
        if checkpoint and args.truncate:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
            job_id = checkpoint["job_id"]
            epoch = checkpoint["epoch"]
        # only use job_id from checkpoint
        elif checkpoint:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
            job_id = checkpoint["job_id"]
        # override job_id and epoch with user arguments if given
        if args.job_id:
            job_id = args.job_id
        if args.max_epoch:
            epoch = args.max_epoch
        if not epoch:
            epoch = float("inf")

        if not job_id:
            out = subprocess.Popen(
                    ['grep "epoch: " ' + trace + ' | grep " job: train"'],
                    shell=True,
                    stdout=subprocess.PIPE
                ).communicate()[0]
            job_id = yaml.load(
                out.decode("utf-8").split("\n")[-2], Loader=yaml.SafeLoader
            ).get("job_id")
        entries = []
        current_job_id = job_id
        job_ids = [current_job_id]
        found_previous = True
        scope_command = ""
        if args.example and args.batch:
            pass
        elif args.example:
            scope_command = " | grep -e 'scope: epoch' -e 'scope: example'"
        elif args.batch:
            scope_command = " | grep -e 'scope: epoch' -e 'scope: batch'"
        else:
            scope_command = " | grep 'scope: epoch'"
        while(found_previous):
            for arg, command in zip(
                    [args.valid, args.test],
                    ["grep -e 'resumed_from_job_id: {}' -e 'parent_job_id: {}' "
                     .format(current_job_id, current_job_id) + trace +
                     " | grep 'job: eval' | grep 'epoch: ' | grep -e 'data: valid' -e 'data: train'"
                     + scope_command,
                     "grep  'resumed_from_job_id: {}' ".format(current_job_id) + trace
                     +" | grep 'epoch: ' | grep 'job: eval'  | grep 'data: test'"
                     + scope_command]
            ):
                if arg:
                    out = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE
                    ).communicate()[0]
                    if len(out):
                        current_entries = [
                            yaml.load(entry, Loader=yaml.SafeLoader)
                            for entry in out.decode("utf-8").split("\n")[0:-1]
                        ]
                        current_entries.extend(entries)
                        entries = current_entries
            # always load train entries to determine the job sequence of 'relevant' jobs
            train_out = subprocess.Popen(
                ['grep  " job_id: {}" '.format(current_job_id) + trace +
                 ' | grep "epoch: " | grep "job: train"' + scope_command],
                shell=True,
                stdout=subprocess.PIPE
            ).communicate()[0]
            if len(train_out):
                current_entries = [
                    yaml.load(entry, Loader=yaml.SafeLoader)
                    for entry in train_out.decode("utf-8").split("\n")[0:-1]
                ]
            resumed_id = current_entries[0].get("resumed_from_job_id")
            if args.train:
                current_entries.extend(entries)
                entries = current_entries
            if resumed_id:
                job_ids.append(resumed_id)
                found_previous = True
                current_job_id = resumed_id
            else:
                found_previous = False
        middle = time.time()
        if args.csv:
            csv_writer = csv.writer(sys.stdout)
            # default attributes from the trace; some have to be treated independently
            # format: Dict[new_name] = original_name
            # change the keys to rename the attribute keys that appear in the csv
            trace_attributes = OrderedDict(
                {"epoch": "epoch",
                 "job": "job",
                 "avg_loss": "avg_loss",
                 "data": "data"}
            )
            # default attributes from config
            # format: Dict[new_name] = original_name
            # change the keys to rename the attribute keys that appear in the csv
            config_attributes = OrderedDict(
                {"valid.metric": "valid.metric",
                 "train.optimizer_args.lr": "train.optimizer_args.lr",
                 "dataset.name": "dataset.name"}
            )
            csv_writer.writerow(
                ["job_id"] +
                [trace_attributes[new_key] for new_key in trace_attributes.keys()] +
                ["split"] + # split column distinguishes train,eval,valid and test entries
                ["valid_metric_value"] +  # name in config value in trace
                [config_attributes[new_key] for new_key in config_attributes.keys()] +
                ["model"] + ["reciprocal"] +
                [key for key in keymap.keys()]
            )
        # store configs for job_id's s.t. they need to be loaded only once
        configs = {}
        for entry in entries:
            if not entry.get("epoch") <= float(epoch):
                continue
            current_job_id = entry.get("job_id")
            if current_job_id in configs.keys():
                config = configs[current_job_id]
            else:
                config = cls.get_config_for_job_id(entry.get("job_id"), folder_path)
                configs[current_job_id] = config
            new_attributes = OrderedDict()
            for new_key in keymap.keys():
                try:
                    new_attributes[new_key] = config.get(keymap[new_key])
                except:
                    # creates empty field if key is not existing
                    new_attributes[new_key] = entry.get(keymap[new_key])
            if args.csv:
                if config.get("model") == "reciprocal_relations_model":
                    model = config.get("reciprocal_relations_model.base_model.type")
                    reciprocal = "Yes"
                else:
                    model = config.get("model")
                    reciprocal = "No"
                split = [""]
                if entry.get("job") == "train":
                    split = ["train"]
                elif entry.get("job") == "eval" and entry.get("resumed_from_job_id"):
                    if entry.get("data") == "test":
                        split = ["test"]
                    elif entry.get("data") == "valid":
                        split = ["eval"]
                elif entry.get("job") == "eval" and entry.get("parent_job_id"):
                    split = ["valid"]
                row_trace = [
                    entry.get(trace_attributes[new_key])
                    for new_key in trace_attributes.keys()
                ]
                row_config = [
                    config.get(config_attributes[new_key])
                    for new_key in config_attributes.keys()
                ]
                csv_writer.writerow(
                    [entry.get("job_id").split("-")[0]] + row_trace + split +
                    [entry.get(config.get("valid.metric"))] + row_config +
                    [model] + [reciprocal] +
                    [new_attributes[new_key] for new_key in new_attributes.keys()]
                )
            else:
                if keymap:
                    entry.update(new_attributes)
                sys.stdout.write(yaml.dump(entry).replace("\n",", "))
                sys.stdout.write("\n")
        end = time.time()
        if args.timeit:
            sys.stdout.write("Grep + processing took {} \n".format(middle - start))
            sys.stdout.write("Writing took {}".format(end - middle))

    @classmethod
    def get_checkpoint_from_path(cls, path):
        if "checkpoint_best.pt" in os.listdir(path):
            return path + "/" + "checkpoint_best.pt"
        else:
            checkpoints =  \
                sorted(
                    list(
                        filter(
                            lambda file: "checkpoint" in file, os.listdir(path)
                        )
                    )
                )

            if len(checkpoints) > 0:
                return path + "/" + checkpoints[-1]
            else:
                print("Nothing was dumped. Did not find a checkpoint in {}".format(path))
                exit()

    @classmethod
    # loading the file raw and not using config.load turned
    # out to be faster and should be sufficient here
    def get_config_for_job_id(cls, job_id, folder_path):
        config = Config(load_default=False)
        config_path = folder_path + "/config/" + job_id.split("-")[0] + ".yaml"
        if os.path.isfile(config_path):
            with open(config_path, "r") as file:
                config.options = yaml.load(file, Loader=yaml.SafeLoader)
        else:
            raise Exception("Could not find config file for job_id")
        return config