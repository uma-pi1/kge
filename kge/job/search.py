import copy
import os
import torch.multiprocessing
import concurrent.futures
from kge.job import Job, Trace
from kge.config import _process_deprecated_options
from kge.util.io import get_checkpoint_file, load_checkpoint


class SearchJob(Job):
    """Base class of jobs for hyperparameter search.

    Provides functionality for scheduling training jobs across workers.
    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        # create data structures for parallel job submission
        self.num_workers = self.config.get("search.num_workers")
        self.device_pool = self.config.get("search.device_pool")
        if len(self.device_pool) == 0:
            self.device_pool = [self.config.get("job.device")]
        if len(self.device_pool) < self.num_workers:
            self.device_pool = self.device_pool * self.num_workers
        self.device_pool = self.device_pool[: self.num_workers]
        self.config.log("Using device pool: {}".format(self.device_pool))
        self.free_devices = copy.deepcopy(self.device_pool)
        self.on_error = self.config.check("search.on_error", ["abort", "continue"])

        self.running_tasks = set()  #: set of futures currently runnning
        self.ready_task_results = list()  #: set of results
        if self.num_workers > 1:
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=torch.multiprocessing.get_context("spawn"),
            )
        else:
            self.process_pool = None  # marks that we run in single process

        self.config.check_range("valid.every", 1, config.get("train.max_epochs"))

        if self.__class__ == SearchJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(config, dataset, parent_job=None):
        """Factory method to create a search job."""

        if config.get("search.type") == "manual":
            from kge.job import ManualSearchJob

            return ManualSearchJob(config, dataset, parent_job)
        elif config.get("search.type") == "grid":
            from kge.job import GridSearchJob

            return GridSearchJob(config, dataset, parent_job)
        elif config.get("search.type") == "ax":
            from kge.job import AxSearchJob

            return AxSearchJob(config, dataset, parent_job)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("search.type")

    def submit_task(self, task, task_arg, wait_when_full=True):
        """Runs the given task with the given argument.

        When ``search.num_workers`` is 1, blocks and runs synchronous. Otherwise,
        schedules the task at a free worker. If no worker is free, either waits
        (`wait_when_full` true) or throws an error (`wait_when_full` false).

        In addition to task_arg, the task is given a keyword argument `device`, holding
        the device on which it should run.

        """
        if self.process_pool is None:
            self.ready_task_results.append(task(task_arg, device=self.free_devices[0]))
        else:
            if len(self.running_tasks) >= self.num_workers:
                if wait_when_full:
                    self.config.log("No more free workers.")
                    self.wait_task()
                else:
                    raise ValueError("no more free workers for running the task")
            task_device = self.free_devices.pop(0)
            future = self.process_pool.submit(task, task_arg, device=task_device)
            future.add_done_callback(lambda _: self.free_devices.append(task_device))
            self.running_tasks.add(future)

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
        del state["process_pool"]
        del state["running_tasks"]
        return state


def _run_train_job(sicnk, device=None):
    """Runs a training job and returns the trace entry of its best validation result.

    Also takes are of appropriate tracing.

    """

    search_job, train_job_index, train_job_config, train_job_count, trace_keys = sicnk

    try:
        # load the job
        if device is not None:
            train_job_config.set("job.device", device)
        search_job.config.log(
            "Starting training job {} ({}/{}) on device {}...".format(
                train_job_config.folder,
                train_job_index + 1,
                train_job_count,
                train_job_config.get("job.device"),
            )
        )
        checkpoint_file = get_checkpoint_file(train_job_config)
        if checkpoint_file is not None:
            checkpoint = load_checkpoint(
                checkpoint_file, train_job_config.get("job.device")
            )
            job = Job.create_from(
                checkpoint=checkpoint,
                new_config=train_job_config,
                dataset=search_job.dataset,
                parent_job=search_job,
            )
        else:
            job = Job.create(
                config=train_job_config,
                dataset=search_job.dataset,
                parent_job=search_job,
            )

        # process the trace entries to far (in case of a resumed job)
        metric_name = search_job.config.get("valid.metric")
        valid_trace = []

        def copy_to_search_trace(job, trace_entry):
            trace_entry = copy.deepcopy(trace_entry)
            for key in trace_keys:
                # Process deprecated options to some extent. Support key renames, but
                # not value renames.
                actual_key = {key: None}
                _process_deprecated_options(actual_key)
                if len(actual_key) > 1:
                    raise KeyError(
                        f"{key} is deprecated but cannot be handled automatically"
                    )
                actual_key = next(iter(actual_key.keys()))
                value = train_job_config.get(actual_key)
                trace_entry[key] = value

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
            search_job.config.log(
                "Skipping running of training job as requested by user."
            )
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
        best["child_job_id"] = best["job_id"]
        del (
            best["job"],
            best["job_id"],
            best["type"],
            best["parent_job_id"],
            best["scope"],
            best["event"],
        )
        search_job.trace(
            event="search_completed",
            echo=True,
            echo_prefix="  ",
            log=True,
            scope="train",
            **best,
        )

        return (train_job_index, best, best_metric)
    except BaseException as e:
        search_job.config.log(
            "Trial {:05d} failed: {}".format(train_job_index, repr(e))
        )
        if search_job.on_error == "continue":
            return (train_job_index, None, None)
        else:
            search_job.config.log(
                "Aborting search due to failure of trial {:05d}".format(train_job_index)
            )
            raise e
