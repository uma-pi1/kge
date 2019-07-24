from kge.job import AutoSearchJob
from kge import Config


class RandomSearchJob(AutoSearchJob):
    """Job for hyperparameter search using random search TODO Website"""

    def _init_(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        # Inititialize Client??

    def init_search(self):
        pass

    def register_trial(self, parameters=None):
        pass

    def register_trial_result(self, trial_id, parameters, trace_entry):
        pass

    def get_best_parameters(self):
        pass

    def run(self):
        pass