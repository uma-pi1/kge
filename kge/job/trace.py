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

    @staticmethod
    def grep_entries(tracefile: str, conjunctions: list, raw=False):
        """ For a given tracefile, returns entries matching patterns with 'grep'.

        :param tracefile: String, path to tracefile
        :param conjunctions: A list of strings(patterns) or tuples with strings to be
        used with grep. Elements of the list denote conjunctions (AND) and
        elements within tuples in the list denote disjunctions (OR). For example,
        conjunctions = [("epoch: 10,", "epoch: 12,"), "job: train"] retrieves all entries
        which are from epoch 10 OR 12 AND belong to training jobs. Arbitrary nested
        combinations of AND and OR are possible.

        :returns: A list of dictionaries containing the matching entries.
        If raw=True returns a list with raw strings of the entries (much faster).

        """
        command = "grep "
        if type(conjunctions[0]) == tuple:
            for disjunction in conjunctions[0]:
                command += "-e '{}' ".format(disjunction)
            command += "{} ".format(tracefile)
        elif type(conjunctions[0]) == str:
            command += "'{}' ".format(conjunctions[0])
            command += "{} ".format(tracefile)
        for el in conjunctions[1:]:
            command += "| grep "
            if type(el) == tuple:
                for disjunction in el:
                    command += "-e '{}' ".format(disjunction)
            elif type(el) == str:
                command += "'{}' ".format(el)
        output = subprocess.Popen(
            [command], shell=True, stdout=subprocess.PIPE
        ).communicate()[0]
        if len(output) and not raw:
            # TODO: if efficiency of dump trace has to be improved:
            # the grep command runs fast also for large trace files
            # the bottleneck is yaml.load() when throughput is large
            entries = [
                yaml.load(entry, Loader=yaml.SafeLoader)
                for entry in output.decode("utf-8").split("\n")[0:-1]
            ]
            return entries
        elif len(output) and raw:
            return output.decode("utf-8").split("\n")[0:-1]
        else:
            return []

    @staticmethod
    def grep_training_trace_entries(
        tracefile: str,
        train: bool,
        test: bool,
        valid: bool,
        example=False,
        batch=False,
        job_id=None,
        epoch_of_last=None,
    ):
        """ Extracts trace entry types from a training job trace.

        For a given job_id the sequence of training job's leading to the job with job_id
        is retrieved. All entry types determined by the options will be included and
        returned as a list of dictionaries. For train entries, all epochs of all job's
        are included. These can be filtered with job_epochs.

        :param tracefile: String
        :param train/test/valid: Boolean whether to include entries of the type
        :param batch/example: Boolean whether to include entries of the scope
        :param job_id: The job_id to determine the end of the training sequence.
        If none, the job_id of the last training entry in the trace is used.
        :param epoch_of_last: The max epoch number the job with job_id is trained.
        Note: all epochs of all training jobs in the sequence are retrieved and can be
        filtered with job_epochs.

        :returns: entries, job_epochs
        entries: list of dictionaries with the respective entries
        job_epochs: a dictionary where the key's are the job id's in the trainig job
        sequence and the values are max epochs numbers the jobs have been trained in
        the sequence.

        """
        if not job_id:
            last_entry = Trace.grep_entries(
                tracefile=tracefile,
                conjunctions=["scope: epoch", "job: train"],
                raw=True,
            )[-1]
            job_id = yaml.load(last_entry, Loader=yaml.SafeLoader).get("job_id")
        if not job_id:
            raise Exception(
                "Could not find a training entry in tracefile."
                "Please check file or specify job_id"
            )
        entries = []
        current_job_id = job_id
        # key is a job and epoch is the max epoch needed for this job
        job_epochs = {current_job_id: epoch_of_last}
        found_previous = True
        scopes = ""
        # scopes is always needed to filter out meta entries
        if example and batch:
            scopes = "scope: epoch", "scope: example", "scope: batch"
        elif example:
            scopes = "scope: epoch", "scope: example"
        elif batch:
            scopes = "scope: epoch", "scope: batch"
        else:
            scopes = "scope: epoch"
        while found_previous:
            # in conj list elements are combined with AND tuple elements with OR
            for arg, conj in zip(
                [valid, test],
                [
                    [
                        (
                            "resumed_from_job_id: {}".format(current_job_id),
                            "parent_job_id: {}".format(current_job_id),
                        ),
                        "job: eval",
                        ("data: valid", "data:train"),
                    ]
                    + [scopes],
                    [
                        (
                            "resumed_from_job_id: {}".format(current_job_id),
                            "parent_job_id: {}".format(current_job_id),
                        ),
                        "job: eval",
                        "data: test",
                    ]
                    + [scopes],
                ],
            ):
                if arg:
                    current_entries = Trace.grep_entries(
                        tracefile=tracefile, conjunctions=conj
                    )
                    if len(current_entries):
                        current_entries.extend(entries)
                        entries = current_entries
            # always load train entries to determine the job sequence of 'relevant' jobs
            current_entries = Trace.grep_entries(
                tracefile=tracefile,
                conjunctions=["job_id: {}".format(current_job_id), "job: train"]
                + [scopes],
            )
            resumed_id = ""
            if len(current_entries):
                resumed_id = current_entries[0].get("resumed_from_job_id")
                if train:
                    current_entries.extend(entries)
                    entries = current_entries
            if resumed_id:
                # used to filter out larger epochs of a previous job
                # from the previous job epochs only up until the current epoch is needed
                # current entries must only contain training type entries
                job_epochs[resumed_id] = current_entries[0].get("epoch") - 1
                found_previous = True
                current_job_id = resumed_id
            else:
                found_previous = False
        return entries, job_epochs


class ObjectDumper:
    @classmethod
    def dump_trace(cls, args):
        start = time.time()
        if not (args.train or args.valid or args.test):
            args.train = True
            args.valid = True
            args.test = True

        checkpoint = None
        if ".pt" in os.path.split(args.source)[-1]:
            checkpoint = args.source
            folder_path = os.path.split(args.source)[0]
        else:
            # determine job_id and epoch from last/best checkpoint automatically
            if args.checkpoint:
                checkpoint = cls.get_checkpoint_from_path(args.source)
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
        if checkpoint and args.truncate:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
            job_id = checkpoint["job_id"]
            epoch = checkpoint["epoch"]
        # only use job_id from checkpoint
        elif checkpoint:
            checkpoint = torch.load(f=checkpoint, map_location="cpu")
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
            job_id=job_id,
            epoch_of_last=epoch
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
                config = cls.get_config_for_job_id(entry.get("job_id"), folder_path)
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

    @classmethod
    def get_checkpoint_from_path(cls, path):
        if "checkpoint_best.pt" in os.listdir(path):
            return os.path.join(path, "checkpoint_best.pt")
        else:
            checkpoints = sorted(
                list(filter(lambda file: "checkpoint" in file, os.listdir(path)))
            )

            if len(checkpoints) > 0:
                return os.path.join(path, checkpoints[-1])
            else:
                print(
                    "Nothing was dumped. Did not find a checkpoint in {}".format(path)
                )
                exit()

    @classmethod
    def get_config_for_job_id(cls, job_id, folder_path):
        config = Config(load_default=True)
        config_path = os.path.join(
            folder_path, "config", job_id.split("-")[0] + ".yaml"
        )
        if os.path.isfile(config_path):
            config.load(config_path, create=True)
        else:
            raise Exception("Could not find config file for job_id")
        return config
