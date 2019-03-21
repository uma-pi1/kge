import yaml
import argparse
import experiment.grid_search as grid_search
import experiment.bayesian_optimization as bayesian_optimization

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--config_file', type=str)
  args = parser.parse_args()

  # load experiment settings
  with open(args.config_file, 'r') as file:
    config = yaml.load(file)
  print(yaml.dump(config))

  # run experiment
  if config['experiment']['type'] == 'grid':
    grid_search.GridSearchExperiment(config)
  elif config['experiment']['type'] == 'bayesian':
    bayesian_optimization.BayesianOptimizationExperiment(config)
  else:
    raise ValueError('Unknown experiment type')
