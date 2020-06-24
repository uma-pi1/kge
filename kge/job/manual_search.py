import copy
from kge import Config, Dataset
from kge.job import SearchJob, Job
import kge.job.search
import concurrent.futures


class ManualSearchJob(SearchJob):
    """Job to perform hyperparameter search for a fixed set of configurations.

    This job creates one subjob (a training job stored in a subfolder) for each
    hyperparameter setting. The training jobs are then run indepedently and results
    analyzed.

    Interrupted searches can be resumed. Subjobs can also be resumed/run directly.
    Configurations can be added/removed/edited by modifying the config file.

    Produces a trace file that contains entries for: each validation performed
    for each job (type=eval), the best validation result of each job
    (type=search, scope=train), and the best overall result (type=search,
    scope=search). Each trace entry contains the values of all relevant
    hyperparameters. To filter just the entries of the last run of this search
    job, use its job_id (note: stored as field parent_job_id in type=eval
    entries).

    """

    def __init__(self, config: Config, dataset: Dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        if self.__class__ == ManualSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def _run(self):
        # read search configurations and expand them to full configs
        search_configs = copy.deepcopy(self.config.get("manual_search.configurations"))
        all_keys = set()
        for i in range(len(search_configs)):
            search_config = search_configs[i]
            folder = search_config["folder"]
            del search_config["folder"]
            config = self.config.clone(folder)
            config.set("job.type", "train")
            config.options.pop("manual_search", None)  # could be large, don't copy
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
        # self.dataset.index("train_sp_to_o")
        # self.dataset.index("train_po_to_s")
        # self.dataset.index("valid_sp_to_o")
        # self.dataset.index("valid_po_to_s")
        # self.dataset.index("test_sp_to_o")
        # self.dataset.index("test_po_to_s")

        # now start running/resuming
        for i, config in enumerate(search_configs):
            task_arg = (self, i, config, len(search_configs), all_keys)
            self.submit_task(kge.job.search._run_train_job, task_arg)
        self.wait_task(concurrent.futures.ALL_COMPLETED)

        # if not running the jobs, stop here
        if not self.config.get("manual_search.run"):
            self.config.log("Skipping evaluation of results as requested by user.")
            return

        # collect results
        best_per_job = [None] * len(search_configs)
        best_metric_per_job = [None] * len(search_configs)
        for ibm in self.ready_task_results:
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
            event="search_completed",
            echo=True,
            echo_prefix="  ",
            log=True,
            scope="search",
            **overall_best
        )
