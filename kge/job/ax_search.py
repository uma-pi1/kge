import torch
import concurrent.futures
from kge.job import SearchJob
from kge import Config
import kge.job.search
from ax.service.ax_client import AxClient
from typing import List

# TODO generalize the code to support other backends than ax
# TODO handle "max_epochs" in some sensible way
# TODO when resuming an experiment, run BO right away (instead of Sobol first)


class AxSearchJob(SearchJob):
    """Job for hyperparameter search using [ax](https://ax.dev/)"""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        self.trial_parameters: List[dict] = []
        self.results: List[dict] = []

    def save(self, filename):
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {"trial_parameters": self.trial_parameters, "results": self.results},
            filename,
        )

    def load(self, filename):
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename)
        self.trial_parameters = checkpoint["trial_parameters"]
        self.results = checkpoint["results"]

    def resume(self):
        last_checkpoint = self.config.last_checkpoint()
        if last_checkpoint is not None:
            checkpoint_file = self.config.checkpoint_file(last_checkpoint)
            self.load(checkpoint_file)
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    def run(self):
        # create experiment
        ax_client = AxClient()
        ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("ax_search.parameters"),
            objective_name="metric_value",
            minimize=False,
        )

        # create worker pool
        if self.config.get("search.num_workers") > 1:
            process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.get("search.num_workers")
            )
            running_trials = set()  # set of trial currently running
            ready_trials = list()  # results of ready trials
        else:
            process_pool = None  # marks that we run in single process

        # let's go
        index_for_trial = []
        trial_no = 0
        max_trials = self.config.get("ax_search.max_trials")
        while trial_no < max_trials:
            self.config.log("Starting trial {}/{}...".format(trial_no, max_trials - 1))

            # determine next trial
            if trial_no >= len(self.trial_parameters):
                # create a new trial
                try:
                    parameters, trial_index = ax_client.get_next_trial()
                    self.trial_parameters.append(parameters)
                    self.results.append(None)
                    index_for_trial.append(trial_index)
                except ValueError:
                    # error: ax needs more data
                    self.config.log(
                        "Cannot generate trial. Will try again after a running trial "
                        + "has completed."
                    )
                    trial_index = None  # marks error
            else:
                # use the trial of the prior run of this job
                parameters = self.trial_parameters[trial_no]
                _, trial_index = ax_client.attach_trial(parameters)
                index_for_trial.append(trial_index)
                self.results.append(None)

            # create job for trial
            if trial_index is not None:
                folder = str("{:05d}".format(trial_no))
                config = self.config.clone(folder)
                config.set("job.type", "train")
                config.set_all(parameters)
                config.init_folder()
            else:
                config = None

            # run or schedule the trial
            if process_pool is None:
                # single process -> just run
                ready_trials = [
                    kge.job.search._run_train_job(
                        (
                            self,
                            trial_no,
                            config,
                            self.config.get("ax_search.max_trials"),
                            parameters.keys(),
                        )
                    )
                ]
            else:
                # multiprocess
                ready_futures = set()  # trials that just become ready

                # wait until there is a free worker for a new trial; also wait when we
                # couldn't generate a new trial since data is lacking
                if (
                    len(running_trials) == self.config.get("search.num_workers")
                    or trial_index is None
                ):
                    self.config.log("Waiting for some trial to complete...")
                    ready_futures, running_trials = concurrent.futures.wait(
                        running_trials, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                # now submit the next trial (in case we had generated one)
                if trial_index is not None:
                    running_trials.add(
                        process_pool.submit(
                            kge.job.search._run_train_job,
                            (
                                self,
                                trial_no,
                                config,
                                self.config.get("ax_search.max_trials"),
                                set(parameters.keys()),
                            ),
                        )
                    )

                    # on last iteration, wait for all running trials to complete
                    if trial_no == max_trials - 1:
                        self.config.log("Waiting for remaining trials to complete...")
                        new_ready_futures, _ = concurrent.futures.wait(
                            running_trials, return_when=concurrent.futures.ALL_COMPLETED
                        )
                        ready_futures = ready_futures.union(new_ready_futures)

                # fetch results of all ready trials
                for future in ready_futures:
                    ready_trials.append(future.result())

            # for each ready trial, store its results
            for ready_trial_no, ready_trial_best, _ in ready_trials:
                self.config.log(
                    "Registering trial {} result: {}".format(
                        ready_trial_no, ready_trial_best["metric_value"]
                    )
                )
                self.results[ready_trial_no] = ready_trial_best

                # register the arm
                # TODO: std dev shouldn't be fixed to 0.0
                ax_client.complete_trial(
                    trial_index=index_for_trial[ready_trial_no],
                    raw_data={"metric_value": (ready_trial_best["metric_value"], 0.0)},
                )

                # save checkpoint
                # TODO make atomic (may corrupt good checkpoint when canceled!)
                self.save(self.config.checkpoint_file(1))

            # clean up
            ready_trials.clear()
            if trial_index is not None:
                # advance to next trial (unless we did not run this one)
                trial_no += 1

        # all done, output best result
        self.config.log("And the winner is...")
        best_parameters, values = ax_client.get_best_parameters()
        self.config.log("best_parameters: {}".format(best_parameters), prefix="  ")
        self.config.log(
            "best_matric_value (estimate): {}".format(values[0]["metric_value"]),
            prefix=" ",
        )

        # record the best result of this job
        self.trace(
            echo=True,
            echo_prefix="  ",
            log=True,
            scope="search",
            metric_value=float(values[0]["metric_value"]),
            **best_parameters
        )
