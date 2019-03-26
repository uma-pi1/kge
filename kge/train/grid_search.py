from kge.experiment.base_experiment import BaseExperiment


class GridSearchExperiment(BaseExperiment):
    """
    Grid search of hyperparameter space
    """

    def __init__(self, config):
        self.param1 = config.param1
