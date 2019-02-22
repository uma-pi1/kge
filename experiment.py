class BaseExperiment:
  """
  Instantiates evaluator, trainer, model, and runs trainer for each experiment setting
  """

  def run(self):
    raise NotImplemented


class GridSearchExperiment(BaseExperiment):
  """
  Grid search of hyperparameter space
  """

  def __init__(self, config):
    self.param1 = config.param1


class BayesianOptimizationExperiment(BaseExperiment):
  """
  'Bayesian' optimization of hyperparameter space
  """
  def __init__(self, config):
    self.param1 = config.param1
