import yaml
import argparse
from kge import job

if __name__ == '__main__':

  # args parsing
  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str)
  args = parser.parse_args()

  # TODO: parse other arugments to override default settings
  # - use argument of form "--key=value", where key is the name of a setting

  # load settings from config file
  with open(args.config_file, 'r') as file:
    config = yaml.load(file)
  print(yaml.dump(config))

  # TODO: override settings with command line args

  # create job
  if config['job']['type'] == 'grid':
    job.grid_search_job.GridSearchExperiment(config)
  elif config['job']['type'] == 'bayesian':
    job.bayesian_optimization_job.BayesianOptimizationExperiment(config)
  else:
    raise ValueError('Unknown experiment type')

  # Run/evaluate job
