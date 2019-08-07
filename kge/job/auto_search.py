import concurrent.futures
from typing import List
import torch
from kge import Config
from kge.job import SearchJob
import kge.job.search

# TODO handle "max_epochs" in some sensible way


class AutoSearchJob(SearchJob):
    """Base class for search jobs that automatically explore the search space.

    Subclasses should implement :func:`init_search`, :func:`register_trial`,
    :func:`register_trial_result`, :func:`get_best_parameters`.

    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        self.num_trials = None # needs to be set in subclasses
        self.trial_ids: List = []  #: backend-specific identifiers for each trial
        self.parameters: List[dict] = []  #: hyper-parameters of each trial
        self.results: List[dict] = []  #: trace entry of best result of each trial

    def load(self, filename):
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename)
        self.parameters = checkpoint["parameters"]
        self.results = checkpoint["results"]

    def save(self, filename):
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save({"parameters": self.parameters, "results": self.results}, filename)

    def resume(self):
        last_checkpoint = self.config.last_checkpoint()
        if last_checkpoint is not None:
            checkpoint_file = self.config.checkpoint_file(last_checkpoint)
            self.load(checkpoint_file)
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    # -- Abstract methods --------------------------------------------------------------

    def init_search(self):
        """Initialize to start a new search experiment."""
        raise NotImplementedError

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
        training job with these parameters.

        """
        raise NotImplementedError

    def get_best_parameters(self):
        "Return a (best parameters, estimated objective value) tuple."
        raise NotImplementedError

    # -- Main --------------------------------------------------------------------------

    def run(self):
        self.init_search()

        # let's go
        trial_no = 0
        while trial_no < self.num_trials:
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
            else:
                # use the trial of a resumed run of this job
                parameters, trial_id = self.register_trial(self.parameters[trial_no])
                self.trial_ids.append(trial_id)

            # create job for trial
            if trial_id is not None:
                folder = str("{:05d}".format(trial_no))
                config = self.config.clone(folder)
                config.set("job.type", "train")
                config.set_all(parameters)
                config.init_folder()

            # run or schedule the trial
            if trial_id is not None:
                self.submit_task(
                    kge.job.search._run_train_job,
                    (
                        self,
                        trial_no,
                        config,
                        self.num_trials,
                        list(parameters.keys()),
                    ),
                )

                # on last iteration, wait for all running trials to complete
                if trial_no == self.num_trials - 1:
                    self.wait_task(return_when=concurrent.futures.ALL_COMPLETED)
            else:
                # couldn't generate a new trial since data is lacking; so wait
                self.wait_task()

            # for each ready trial, store its results
            for ready_trial_no, ready_trial_best, _ in self.ready_task_results:
                self.config.log(
                    "Registering trial {} result: {}".format(
                        ready_trial_no, ready_trial_best["metric_value"]
                    )
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

        # all done, output best trial result
        trial_metric_values = list(
            map(lambda trial_best: trial_best["metric_value"], self.results)
        )
        best_trial_index = trial_metric_values.index(max(trial_metric_values))
        metric_name = self.results[best_trial_index]["metric_name"]
        self.config.log(
            "Best trial: {}={}".format(
                metric_name, trial_metric_values[best_trial_index]
            )
        )
        self.trace(echo=True,
                   echo_prefix="  ",
                   log=True,
                   scope="search",
                   **self.results[best_trial_index])

        # DISABLED FOR NOW SINCE IDENTICAL TO BEST TRIAL
        # output parameter estimates
        # best_parameters, best_value_estimate = self.get_best_parameters()
        # self.config.log(
        #     "Search result (estimate): {}={}".format(metric_name, best_value_estimate)
        # )
        # self.config.log("parameters: {}".format(best_parameters), prefix="  ")
