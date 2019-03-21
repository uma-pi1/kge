from kge.experiment.base_experiment import BaseExperiment


class BayesianOptimizationExperiment(BaseExperiment):
  """
  'Bayesian' optimization of hyperparameter space
  """
  def __init__(self, config):
    self.param1 = config.param1
