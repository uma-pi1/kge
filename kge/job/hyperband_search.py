from kge.job import AutoSearchJob
from kge import Config


class HyperbandSearchJob(AutoSearchJob):
    """Job for hyperparameter search using hyperband."""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

    def init_search(self):


    def register_trial(self, parameters=None):


    def register_trial_result(self, trial_id, parameters, trace_entry):


    def get_best_parameters(self):

