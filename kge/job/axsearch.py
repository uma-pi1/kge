import torch
from kge.job import Job
from kge import Config
from kge.job.search import _run_train_job
from ax.service.ax_client import AxClient
from typing import List


# TODO generalize the code to support other backends than ax
# TODO handle "max_epochs" in some sensible way
# TODO support running of multiple trials in parallel

class AxSearchJob(Job):
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
        ax_client = AxClient()
        ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("axsearch.parameters"),
            objective_name="metric_value",
            minimize=False,
        )

        max_trials = self.config.get("axsearch.max_trials")
        for trial_no in range(max_trials):
            self.config.log("Starting trial {}/{}...".format(trial_no, max_trials))

            # determine next trial
            if trial_no >= len(self.trial_parameters):
                # create a new trial
                parameters, trial_index = ax_client.get_next_trial()
                self.trial_parameters.append(parameters)
            else:
                # use the trial of the prior run of this job
                parameters = self.trial_parameters[trial_no]
                ax_client.attach_trial(parameters)
            self.config.log("Parameters: {}".format(parameters), prefix="  ")

            # evaluate the trial
            folder = str("{:05d}".format(trial_no))
            config = self.config.clone(folder)
            config.set("job.type", "train")
            config.set_all(parameters)
            config.init_folder()

            # run it
            _, best, _ = _run_train_job(
                (
                    self,
                    trial_no,
                    config,
                    self.config.get("axsearch.max_trials"),
                    parameters.keys(),
                )
            )

            # remember it
            self.results.append(best)
            self.config.log("Result: {}".format(best["metric_value"]), prefix="  ")

            # register the arm
            # TODO: std dev shouldn't be fixed to 0.0
            ax_client.complete_trial(
                trial_index=trial_no,
                raw_data={"metric_value": (best["metric_value"], 0.0)},
            )

            # save checkpoint if necessary
            if trial_no >= len(self.results) - 1:
                self.save(self.config.checkpoint_file(trial_no))

        # all done, output best result
        best_parameters, values = ax_client.get_best_parameters()
        self.config.log("best_parameters: {}".format(best_parameters))
        self.config.log("values: {}".format(values[0]))

        # record the best result of this job
        self.trace(
            echo=True,
            echo_prefix="  ",
            log=True,
            scope="search",
            metric_value=float(values[0]["metric_value"]),
            **best_parameters
        )
