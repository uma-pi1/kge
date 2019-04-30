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

    Interrupted searches can be resumed. Subjobs can also be resumed/run
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

        # now start running/resuming
        metric_name = self.config.get("valid.metric")

        def run_job(i_config):
            i, config = i_config
            # load the job
            self.config.log(
                "Starting training job {} ({}/{})...".format(
                    config.folder, i + 1, len(search_configs)
                )
            )
            job = Job.create(config, self.dataset, parent_job=self)
            job.resume()

            # process the trace entries to far (in case of a resumed job)
            valid_trace = []

            def copy_to_search_trace(job, trace_entry):
                trace_entry = copy.deepcopy(trace_entry)
                for key in all_keys:
                    trace_entry[key] = config.get(key)

                trace_entry["folder"] = os.path.split(config.folder)[1]
                metric_value = Trace.get_metric(trace_entry, metric_name)
                trace_entry["metric_name"] = metric_name
                trace_entry["metric_value"] = metric_value
                trace_entry["parent_job_id"] = self.job_id
                self.config.trace(**trace_entry)
                valid_trace.append(trace_entry)

            for trace_entry in job.valid_trace:
                copy_to_search_trace(None, trace_entry)

            # run the job (adding new trace entries as we go)
            if self.config.get("search.run"):
                job.after_valid_hooks.append(copy_to_search_trace)
                job.run()
            else:
                self.config.log(
                    "Skipping running of training job as requested by user."
                )

            # analyze the result
            self.config.log("Best result in this training job:")
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
            self.trace(echo=True, echo_prefix="  ", log=True, scope="train", **best)
            return (i, best, best_metric)

        # and go
        if self.config.get("search.num_workers") == 1:
            result = list(map(run_job, enumerate(search_configs)))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.get("search.num_workers")
            ) as e:
                result = list(
                    e.map(run_job, enumerate(search_configs))
                )

        # collect results
        best_per_job = [None] * len(search_configs)
        best_metric_per_job = [None] * len(search_configs)
        for ibm in result:
            i, best, best_metric = ibm
            best_per_job[i] = best
            best_metric_per_job[i] = best_metric

        # produce an overall summary
        self.config.log("Result summary:")
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
