from kge.job import AutoSearchJob
from kge import Config
from ax.service.ax_client import AxClient
from typing import List

# TODO when resuming an experiment, run BO right away (instead of Sobol first)


class AxSearchJob(AutoSearchJob):
    """Job for hyperparameter search using [ax](https://ax.dev/)."""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.ax_client = None

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = super(AxSearchJob, self).__getstate__()
        del state['ax_client']
        return state

    def init_search(self):
        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            name=self.job_id,
            parameters=self.config.get("ax_search.parameters"),
            objective_name="metric_value",
            minimize=False,
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
