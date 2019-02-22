class Trainer:
  """
  Takes config object, model and evaluator
  Instantiates dataview
  Trains model (knows which forward function to use)
  """

  def __init__(self, config, model, evaluation):
    self.param1 = config.param1
    self.model = model