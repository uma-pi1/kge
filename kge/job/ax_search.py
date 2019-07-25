import concurrent.futures
from kge.job import AutoSearchJob
from kge import Config
import kge.job.search
from ax.service.ax_client import AxClient
from typing import List


class AxSearchJob(AutoSearchJob):
    """Job for hyperparameter search using [ax](https://ax.dev/)."""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.ax_client = None

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = super(AxSearchJob, self).__getstate__()
        del state["ax_client"]
        return state

    def init_search(self):
        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("ax_search.parameters"),
            objective_name="metric_value",
            minimize=False,
        )

        # By default, ax first uses a Sobol strategy for a certain number of arms,
        # followed by Bayesian Optimization. If we resume this job, some of the Sobol
        # arms may have already been generated. The corresponding arms will be
        # registered later (when this job's run method is executed), but here we already
        # change the generation strategy to take account of these configurations.
        num_generated = len(self.parameters)
        if num_generated > 0:
            old_curr = self.ax_client.generation_strategy._curr
            new_num_arms = max(0, old_curr.num_arms - num_generated)
            new_curr = old_curr._replace(num_arms=new_num_arms)
            self.ax_client.generation_strategy._curr = new_curr
            self.config.log(
                "Reduced number of arms for first generation step of "
                + "ax_client from {} to {} due to prior data.".format(
                    old_curr.num_arms, new_curr.num_arms
                )
            )

    def register_trial(self, parameters=None):
        try:
            if parameters is None:
                parameters, trial_id = self.ax_client.get_next_trial()
            else:
                _, trial_id = self.ax_client.attach_trial(parameters)
            return parameters, trial_id
        except ValueError:
            # error: ax needs more data
            return None, None

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # TODO: std dev shouldn't be fixed to 0.0
        self.ax_client.complete_trial(
            trial_index=trial_id,
            raw_data={"metric_value": (trace_entry["metric_value"], 0.0)},
        )

    def get_best_parameters(self):
        best_parameters, values = self.ax_client.get_best_parameters()
        return best_parameters, float(values[0]["metric_value"])

    def run(self):
        self.init_search()

        # let's go
        trial_no = 0
        num_trials = self.config.get("ax_search.num_trials")
        while trial_no < num_trials:
            self.config.log(
                "Registering trial {}/{}...".format(trial_no, num_trials - 1)
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

                import sys
                print(parameters.keys())

                self.submit_task(
                    kge.job.search._run_train_job,
                    (
                        self,
                        trial_no,
                        config,
                        self.config.get("ax_search.num_trials"),
                        list(parameters.keys()),
                    ),
                )

                # on last iteration, wait for all running trials to complete
                if trial_no == num_trials - 1:
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
