import copy
import os
import concurrent.futures
from kge.job import Job, Trace
from kge import Config


class SearchJob(Job):
    """Job to perform hyperparameter search.

    This job creates one subjob (a training job stored in a subfolder) for each
    hyperparameter setting. The training jobs are then run in sequence and
    results analyzed.

ikljZZ    Interrupted searches can be resumed. Subjobs can also be resumed/run
    directly. Configurations can be added/removed by modifying the config file.

    Produces a trace file that contains entries for: each validation performed
    for each job (type=eval), the best validation result of each job
    (type=search, scope=train), and the best overall result (type=search,
    scope=search). Each trace entry contains the values of all relevant
    hyperparameters. To filter just the entries of the last run of this search
    job, use its job_id (note: stored as field parent_job_id in type=eval
    entries).

    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

    def resume(self):
        # no need to do anything here; run code automatically resumes
        pass

    def run(self):
        # read search configurations and expand them to full configs
        search_configs = copy.deepcopy(self.config.get("search.configurations"))
        all_keys = set()
        for i in range(len(search_configs)):
            search_config = search_configs[i]
            folder = search_config["folder"]
            del search_config["folder"]
            config = self.config.clone(folder)
            config.set("job.type", "train")
            flattened_search_config = Config.flatten(search_config)
            config.set_all(flattened_search_config)
            all_keys.update(flattened_search_config.keys())
            search_configs[i] = config

        # create folders for search configs (existing folders remain
        # unmodified)
        for config in search_configs:
            config.init_folder()

        # TODO find a way to create all indexes before running the jobs. The quick hack
        # below does not work becuase pytorch then throws a "too many open files" error
        # self.dataset.index_1toN("train", "sp")
        # self.dataset.index_1toN("train", "po")
        # self.dataset.index_1toN("valid", "sp")
        # self.dataset.index_1toN("valid", "po")
        # self.dataset.index_1toN("test", "sp")
        # self.dataset.index_1toN("test", "po")

        # now start running/resuming
        tasks = [
            (self, i, config, len(search_configs), all_keys)
            for i, config in enumerate(search_configs)
        ]
        if self.config.get("search.num_workers") == 1:
            result = list(map(_run_train_job, tasks))
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.get("search.num_workers")
            ) as e:
                result = list(e.map(_run_train_job, tasks))

        # collect results
        best_per_job = [None] * len(search_configs)
        best_metric_per_job = [None] * len(search_configs)
        for ibm in result:
            i, best, best_metric = ibm
            best_per_job[i] = best
            best_metric_per_job[i] = best_metric

        # produce an overall summary
        self.config.log("Result summary:")
        metric_name = self.config.get("valid.metric")
        overall_best = None
        overall_best_metric = None
        for i in range(len(search_configs)):
            best = best_per_job[i]
            best_metric = best_metric_per_job[i]
            if not overall_best or overall_best_metric < best_metric:
                overall_best = best
                overall_best_metric = best_metric
            self.config.log(
                "{}={:.3f} after {} epochs in folder {}".format(
                    metric_name, best_metric, best["epoch"], best["folder"]
                ),
                prefix="  ",
            )
        self.config.log("And the winner is:")
        self.config.log(
            "{}={:.3f} after {} epochs in folder {}".format(
                metric_name,
                overall_best_metric,
                overall_best["epoch"],
                overall_best["folder"],
            ),
            prefix="  ",
        )
        self.config.log("Best overall result:")
        self.trace(
            echo=True, echo_prefix="  ", log=True, scope="search", **overall_best
        )


def _run_train_job(sicnk):
    search_job, i, config, n, all_keys = sicnk

    # load the job
    search_job.config.log(
        "Starting training job {} ({}/{})...".format(config.folder, i + 1, n)
    )
    job = Job.create(config, search_job.dataset, parent_job=search_job)
    job.resume()

    # process the trace entries to far (in case of a resumed job)
    metric_name = search_job.config.get("valid.metric")
    valid_trace = []

    def copy_to_search_trace(job, trace_entry):
        trace_entry = copy.deepcopy(trace_entry)
        for key in all_keys:
            trace_entry[key] = config.get(key)

        trace_entry["folder"] = os.path.split(config.folder)[1]
        metric_value = Trace.get_metric(trace_entry, metric_name)
        trace_entry["metric_name"] = metric_name
        trace_entry["metric_value"] = metric_value
        trace_entry["parent_job_id"] = search_job.job_id
        search_job.config.trace(**trace_entry)
        valid_trace.append(trace_entry)

    for trace_entry in job.valid_trace:
        copy_to_search_trace(None, trace_entry)

    # run the job (adding new trace entries as we go)
    if search_job.config.get("search.run"):
        job.after_valid_hooks.append(copy_to_search_trace)
        job.run()
    else:
        search_job.config.log("Skipping running of training job as requested by user.")

    # analyze the result
    search_job.config.log("Best result in this training job:")
    best = None
    best_metric = None
    for trace_entry in valid_trace:
        metric = trace_entry["metric_value"]
        if not best or best_metric < metric:
            best = trace_entry
            best_metric = metric

    # record the best result of this job
    del (
        best["job"],
        best["job_id"],
        best["type"],
        best["parent_job_id"],
        best["scope"],
    )
    search_job.trace(echo=True, echo_prefix="  ", log=True, scope="train", **best)
    return (i, best, best_metric)
