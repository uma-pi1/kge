import copy
import os
import concurrent.futures
from kge.job import Job, Trace


class SearchJob(Job):
    """Base class of jobs for hyperparameter search.

    Provides functionality for scheduling training jobs across workers.
    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        # create data structures for parallel job submission
        self.num_workers = self.config.get("search.num_workers")
        self.running_tasks = set()  #: set of futures currently runnning
        self.ready_task_results = list()  #: set of results
        if self.num_workers > 1:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers
            )
        else:
            self.process_pool = None  # marks that we run in single process

    def create(config, dataset, parent_job=None):
        """Factory method to create a search job."""

        if config.get("search.type") == "manual":
            from kge.job import ManualSearchJob
            return ManualSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "random":
            from kge.job import RandomSearchJob
            return RandomSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "grid":
            from kge.job import GridSearchJob
            return GridSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "ax":
            from kge.job import AxSearchJob
            return AxSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "hyperband":
            from kge.job import HyperBandSearchJob
            return HyperBandSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "bohb":
            from kge.job import BOHBSearchJob
            return BOHBSearchJob(config, dataset, parent_job)

        elif config.get("search.type") == "tpe":
            from kge.job import TPESearchJob
            return TPESearchJob(config, dataset, parent_job)

        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("search.type")

    def submit_task(self, task, task_arg, wait_when_full=True):
        """Runs the given task with the given argument.

        When ``search.num_workers`` is 1, blocks and runs synchronous. Otherwise,
        schedules the task at a free worker. If no worker is free, either waits
        (`wait_when_full` true) or throws an error (`wait_when_full` false).

        """
        if self.process_pool is None:
            self.ready_task_results.append(task(task_arg))
        else:
            if len(self.running_tasks) >= self.num_workers:
                if wait_when_full:
                    self.config.log("No more free workers.")
                    self.wait_task()
                else:
                    raise ValueError("no more free workers for running the task")
            self.running_tasks.add(self.process_pool.submit(task, task_arg))

    def wait_task(self, return_when=concurrent.futures.FIRST_COMPLETED):
        """Waits for one or more running tasks to complete.

        Results of all completed tasks are copied into ``self.ready_task_results``.

        When no task is running, does nothing.

        """
        if len(self.running_tasks) > 0:
            self.config.log("Waiting for tasks to complete...")
            ready_tasks, self.running_tasks = concurrent.futures.wait(
                self.running_tasks, return_when=return_when
            )
            for task in ready_tasks:
                self.ready_task_results.append(task.result())

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = dict(self.__dict__)
        del state['process_pool']
        del state['running_tasks']
        return state


# TODO add job submission (to devices/cpus) etc. to SearchJob main class with a simpler
# API
def _run_train_job(sicnk):
    """Runs a training job and returns the trace entry of its best validation result.

    Also takes are of appropriate tracing.

    """

    search_job, train_job_index, train_job_config, train_job_count, trace_keys = sicnk

    # load the job
    search_job.config.log(
        "Starting training job {} ({}/{})...".format(
            train_job_config.folder, train_job_index + 1, train_job_count
        )
    )
    job = Job.create(train_job_config, search_job.dataset, parent_job=search_job)
    job.resume()

    # process the trace entries to far (in case of a resumed job)
    metric_name = search_job.config.get("valid.metric")
    valid_trace = []

    def copy_to_search_trace(job, trace_entry):
        trace_entry = copy.deepcopy(trace_entry)
        for key in trace_keys:
            trace_entry[key] = train_job_config.get(key)

        trace_entry["folder"] = os.path.split(train_job_config.folder)[1]
        metric_value = Trace.get_metric(trace_entry, metric_name)
        trace_entry["metric_name"] = metric_name
        trace_entry["metric_value"] = metric_value
        trace_entry["parent_job_id"] = search_job.job_id
        search_job.config.trace(**trace_entry)
        valid_trace.append(trace_entry)

    for trace_entry in job.valid_trace:
        copy_to_search_trace(None, trace_entry)

    # run the job (adding new trace entries as we go)
    # TODO make this less hacky (easier once integrated into SearchJob)
    from kge.job import ManualSearchJob

    if not isinstance(search_job, ManualSearchJob) or search_job.config.get(
        "manual_search.run"
    ):
        job.post_valid_hooks.append(copy_to_search_trace)
        job.run()
    else:
        search_job.config.log("Skipping running of training job as requested by user.")
        return (train_job_index, None, None)

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
    return (train_job_index, best, best_metric)
