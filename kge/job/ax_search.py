from math import ceil

from ax import Models
from ax.core import ObservationFeatures
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy

from kge.job import AutoSearchJob, Job
from kge import Config
from ax.service.ax_client import AxClient
from typing import List


class AxSearchJob(AutoSearchJob):
    """Job for hyperparameter search using [ax](https://ax.dev/)."""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.num_trials = self.config.get("ax_search.num_trials")
        self.num_sobol_trials = self.config.get("ax_search.num_sobol_trials")
        self.ax_client: AxClient = None

        if self.__class__ == AxSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = super(AxSearchJob, self).__getstate__()
        del state["ax_client"]
        return state

    def _prepare(self):
        super()._prepare()
        if self.num_sobol_trials > 0:
            # BEGIN: from /ax/service/utils/dispatch.py
            generation_strategy = GenerationStrategy(
                name="Sobol+GPEI",
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=self.num_sobol_trials,
                        min_trials_observed=ceil(self.num_sobol_trials / 2),
                        enforce_num_trials=True,
                        model_kwargs={"seed": self.config.get("ax_search.sobol_seed")},
                    ),
                    GenerationStep(model=Models.GPEI, num_trials=-1, max_parallelism=3),
                ],
            )
            # END: from /ax/service/utils/dispatch.py

            self.ax_client = AxClient(generation_strategy=generation_strategy)
            choose_generation_strategy_kwargs = dict()
        else:
            self.ax_client = AxClient()
            # set random_seed that will be used by auto created sobol search from ax
            # note that here the argument is called "random_seed" not "seed"
            choose_generation_strategy_kwargs = {
                "random_seed": self.config.get("ax_search.sobol_seed")
            }
        self.ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("ax_search.parameters"),
            objective_name="metric_value",
            minimize=False,
            parameter_constraints=self.config.get("ax_search.parameter_constraints"),
            choose_generation_strategy_kwargs=choose_generation_strategy_kwargs,
        )
        self.config.log(
            "ax search initialized with {}".format(self.ax_client.generation_strategy)
        )

        # Make sure sobol models are resumed correctly
        if self.ax_client.generation_strategy._curr.model == Models.SOBOL:

            self.ax_client.generation_strategy._set_current_model(
                experiment=self.ax_client.experiment, data=None
            )

            # Regenerate and drop SOBOL arms already generated. Since we fixed the seed,
            # we will skip exactly the arms already generated in the job being resumed.
            num_generated = len(self.parameters)
            if num_generated > 0:
                num_sobol_generated = min(
                    self.ax_client.generation_strategy._curr.num_trials, num_generated
                )
                for i in range(num_sobol_generated):
                    generator_run = self.ax_client.generation_strategy.gen(
                        experiment=self.ax_client.experiment
                    )
                    # self.config.log("Skipped parameters: {}".format(generator_run.arms))
                self.config.log(
                    "Skipped {} of {} Sobol trials due to prior data.".format(
                        num_sobol_generated,
                        self.ax_client.generation_strategy._curr.num_trials,
                    )
                )

    def register_trial(self, parameters=None):
        trial_id = None
        try:
            if parameters is None:
                parameters, trial_id = self.ax_client.get_next_trial()
            else:
                _, trial_id = self.ax_client.attach_trial(parameters)
        except Exception as e:
            self.config.log(
                "Cannot generate trial parameters. Will try again after a "
                + "running trial has completed. message was: {}".format(e)
            )
        return parameters, trial_id

    def register_trial_result(self, trial_id, parameters, trace_entry):
        if trace_entry is None:
            self.ax_client.log_trial_failure(trial_index=trial_id)
        else:
            self.ax_client.complete_trial(
                trial_index=trial_id, raw_data=trace_entry["metric_value"]
            )

    def get_best_parameters(self):
        best_parameters, values = self.ax_client.get_best_parameters()
        return best_parameters, float(values[0]["metric_value"])
