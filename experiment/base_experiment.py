class BaseExperiment:
  """
  Instantiates evaluator, trainer, model, and runs trainer for each experiment setting
  """

  def run(self):
    raise NotImplementedError
