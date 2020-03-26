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


from kge.misc import kge_base_dir
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
        """For a given tracefile, return entries that match patterns with 'grep'.

        :param tracefile: String, path to tracefile
        :param conjunctions: A list of strings(patterns) or tuples with strings to be
        used with grep. Elements of the list denote conjunctions (AND) and
        elements within tuples in the list denote disjunctions (OR). For example,
        conjunctions = [("epoch: 10,", "epoch: 12,"), "job: train"] retrieves all
        entries which are from epoch 10 OR 12 AND belong to training jobs.

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
        """Extract trace entry types from a training job trace.

        For a given job_id, the sequence of training job's leading to the job with
        job_id is retrieved. All entry types specified by the options that are
        associated with these jobs will be included and returned as a list of
        dictionaries. For train entries, all epochs of all job's are included. These can
        be filtered with job_epochs.

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
        job_epochs: a dictionary where the key's are the job id's in the training job
        sequence and the values are the max epochs numbers the jobs have been trained in
        the sequence.

        """
        if not job_id:
            entries = Trace.grep_entries(
                tracefile=tracefile,
                conjunctions=["scope: epoch", "job: train"],
                raw=True,
            )
            if not entries:
                return [], dict()
            job_id = yaml.load(entries[-1], Loader=yaml.SafeLoader).get("job_id")
        if not job_id:
            raise Exception(
                "Could not find a training entry in tracefile."
                "Please check file or specify job_id"
            )
        entries = []
        current_job_id = job_id
        # key is a job and epoch is the max epoch needed for this job
        job_epochs = {}
        added_last = False
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
                            " resumed_from_job_id: {}".format(current_job_id),
                            " parent_job_id: {}".format(current_job_id),
                        ),
                        " job: eval",
                        (
                            " split: valid",
                            " split: train",
                            # old keys
                            " data: valid",
                            " data: train",
                        ),
                    ]
                    + [scopes],
                    [
                        (
                            " resumed_from_job_id: {}".format(current_job_id),
                            " parent_job_id: {}".format(current_job_id),
                        ),
                        " job: eval",
                        (" split: test", " data: test"),
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
                conjunctions=[" job_id: {}".format(current_job_id), " job: train"]
                + [scopes],
            )
            resumed_id = ""
            if len(current_entries):
                if not added_last:
                    job_epochs[current_entries[0].get("job_id")] = epoch_of_last
                    added_last = True
                resumed_id = current_entries[0].get("resumed_from_job_id")
                if train:
                    current_entries.extend(entries)
                    entries = current_entries
            if resumed_id:
                # used to filter out larger epochs of a previous job
                # from the previous job epochs only until the current epoch are needed
                # current entries must only contain training type entries
                job_epochs[resumed_id] = current_entries[0].get("epoch") - 1
                found_previous = True
                current_job_id = resumed_id
            else:
                found_previous = False
        return entries, job_epochs

    @staticmethod
    def grep_trace_entries(tracefile: str, job, scope, job_id=None):
        "All trace entries for the specified job_id or, if unspecified, the last job_id"
        if not job_id:
            entries = Trace.grep_entries(
                tracefile=tracefile,
                conjunctions=[f"job: {job}", f"scope: {scope}"],
                raw=True,
            )
            if not entries:
                return [], dict()
            job_id = yaml.load(entries[-1], Loader=yaml.SafeLoader).get("job_id")

        entries = Trace.grep_entries(
            tracefile=tracefile, conjunctions=[f"job_id: {job_id}", f"scope: {scope}"],
        )
        return entries
