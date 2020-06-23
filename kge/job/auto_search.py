import concurrent.futures
from typing import List
import torch
from kge import Config
from kge.config import _process_deprecated_options
from kge.job import SearchJob, Job
import kge.job.search
import copy
import gc

# TODO handle "max_epochs" in some sensible way


class AutoSearchJob(SearchJob):
    """Base class for search jobs that automatically explore the search space.

    Subclasses should implement :func:`init_search`, :func:`register_trial`,
    :func:`register_trial_result`, :func:`get_best_parameters`.

    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        self.num_trials = None  # needs to be set in subclasses
        self.trial_ids: List = []  #: backend-specific identifiers for each trial
        self.parameters: List[dict] = []  #: hyper-parameters of each trial
        self.results: List[dict] = []  #: trace entry of best result of each trial

        if self.__class__ == AutoSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def save(self, filename):
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "type": "search",
                "parameters": self.parameters,
                "results": self.results,
                "job_id": self.job_id,
            },
            filename,
        )

    def _load(self, checkpoint):
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.parameters = checkpoint["parameters"]
        self.results = checkpoint["results"]
        self.trace(
            event="job_resumed", checkpoint_file=checkpoint["file"]
        )
        self.config.log(
            "Resuming search from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    # -- Abstract methods --------------------------------------------------------------

    def register_trial(self, parameters=None):
        """Start a new trial.

        If parameters is ``None``, automatically determine a suitable set of parameter
        values. Otherwise, use the specified parameters.

        Return a (parameters, trial id)-tuple or ``(None, None)``. The trial id is not
        further interpreted, but can be used to store metadata for this trial.
        ``(None,None)`` is returned only if parameters is ``None`` and there are

        registered trials for which the result has not yet been registered via
        :func:`register_trial_results`.

        """
        raise NotImplementedError

    def register_trial_result(self, trial_id, parameters, trace_entry):
        """Register the result of a trial.

        `trial_id` and `parameters` should have been produced by :func:`register_trial`.
        `trace_entry` should be the trace entry for the best validation result of a
        training job with these parameters. If `trace_entry` is `None`, the trial
        is treated as failed.

        """
        raise NotImplementedError

    def get_best_parameters(self):
        "Return a (best parameters, estimated objective value) tuple."
        raise NotImplementedError

    # -- Main --------------------------------------------------------------------------

    def _run(self):

        # let's go
        trial_no = 0
        while trial_no < self.num_trials:
            gc.collect()
            self.config.log(
                "Registering trial {}/{}...".format(trial_no, self.num_trials - 1)
            )

            # determine next trial
            if trial_no >= len(self.parameters):
                # create a new trial
                parameters, trial_id = self.register_trial()
                if trial_id is None:
                    self.config.log(
                        "Cannot generate trial parameters. Will try again after a "
                        + "running trial has completed."
                    )
                else:
                    # remember the trial
                    self.trial_ids.append(trial_id)
                    self.parameters.append(parameters)
                    self.results.append(None)
                    self.config.log(
                        "Created trial {:05d} with parameters: {}".format(
                            trial_no, parameters
                        )
                    )
            else:
                # use the trial of a resumed run of this job
                parameters, trial_id = self.register_trial(self.parameters[trial_no])
                self.trial_ids.append(trial_id)
                self.config.log(
                    "Resumed trial {:05d} with parameters: {}".format(
                        trial_no, parameters
                    )
                )

            if trial_id is None:
                # couldn't generate a new trial since data is lacking; so wait for data
                self.wait_task()
            elif self.results[trial_no] is not None:
                # trial result is in checkpoint, use it (from prior run of this job)
                self.config.log(
                    "Registering trial {:05d} result: {}".format(
                        trial_no, self.results[trial_no]
                    )
                )
                self.register_trial_result(
                    self.trial_ids[trial_no],
                    self.parameters[trial_no],
                    self.results[trial_no],
                )
            else:  # trial_id is valid, but no result yet
                # create/resume job for trial
                folder = str("{:05d}".format(trial_no))
                config = self.config.clone(folder)
                config.set("job.type", "train")
                config.set_all(_process_deprecated_options(copy.deepcopy(parameters)))
                config.init_folder()

                # save checkpoint here so that trial is not lost
                # TODO make atomic (may corrupt good checkpoint when canceled!)
                self.save(self.config.checkpoint_file(1))

                # run or schedule the trial
                self.submit_task(
                    kge.job.search._run_train_job,
                    (self, trial_no, config, self.num_trials, list(parameters.keys())),
                )

            # on last iteration, wait for all running trials to complete
            if trial_id is not None and trial_no == self.num_trials - 1:
                self.wait_task(return_when=concurrent.futures.ALL_COMPLETED)

            # for each ready trial, store its results
            for ready_trial_no, ready_trial_best, _ in self.ready_task_results:
                if ready_trial_best is not None:
                    self.config.log(
                        "Registering trial {:05d} result: {}".format(
                            ready_trial_no, ready_trial_best["metric_value"]
                        )
                    )
                else:
                    # TODO: currently cannot distinguish failed trials from trials that
                    # haven't been run to completion. Both will have their entry in
                    # self.results set to None
                    self.config.log(
                        "Registering failed trial {:05d}".format(ready_trial_no)
                    )
                self.results[ready_trial_no] = ready_trial_best
                self.register_trial_result(
                    self.trial_ids[ready_trial_no],
                    self.parameters[ready_trial_no],
                    ready_trial_best,
                )

                # save checkpoint
                # TODO make atomic (may corrupt good checkpoint when canceled!)
                self.save(self.config.checkpoint_file(1))

            # clean up
            self.ready_task_results.clear()
            if trial_id is not None:
                # advance to next trial (unless we did not run this one)
                trial_no += 1

        # all done, output failed trials result
        failed_trials = [i for i in range(len(self.results)) if self.results[i] is None]
        self.config.log(
            "{} trials were successful, {} trials failed".format(
                len(self.results) - len(failed_trials), len(failed_trials)
            )
        )
        if len(failed_trials) > 0:
            self.config.log(
                "Failed trials: {}".format(
                    " ".join(["{:05d}".format(x) for x in failed_trials])
                )
            )

        # and best trial
        if len(failed_trials) != len(self.results):
            trial_metric_values = [
                float("-Inf") if result is None else result["metric_value"]
                for result in self.results
            ]
            best_trial_index = trial_metric_values.index(max(trial_metric_values))
            metric_name = self.results[best_trial_index]["metric_name"]
            self.config.log(
                "Best trial ({:05d}): {}={}".format(
                    best_trial_index, metric_name, trial_metric_values[best_trial_index]
                )
            )

            self.trace(
                even="search_completed",
                echo=True,
                echo_prefix="  ",
                log=True,
                scope="search",
                **self.results[best_trial_index]
            )

        # DISABLED FOR NOW SINCE IDENTICAL TO BEST TRIAL
        # output parameter estimates
        # best_parameters, best_value_estimate = self.get_best_parameters()
        # self.config.log(
        #     "Search result (estimate): {}={}".format(metric_name, best_value_estimate)
        # )
        # self.config.log("parameters: {}".format(best_parameters), prefix="  ")
