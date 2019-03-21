import datetime
import yaml
import argparse
from kge import job

if __name__ == '__main__':
  # load default config file
  with open('kge/default.yaml', 'r') as file:
    config = yaml.load(file)

  # parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  # TODO add all entries of config as arguments to parser
  args = parser.parse_args()

  # load user config file (overwrites defaults)
  if args.config is not None:
    with open(args.config, 'r') as file:
      user_config = yaml.load(file)
      # TODO deep merge into configs

  # TODO override with command line arguments

  # validate arguments and set defaults
  if config['output']['folder'] == '':
    config['output']['folder'] = \
      datetime.datetime.now().strftime("%Y%m%d-%H%M") \
      + "-" + config['dataset']['name'] \
      + "-" + config['model']['name']

  # print status information
  # TODO nicer
  print(yaml.dump(config))

  # create job
  # if config['job']['type'] == 'grid':
  #   job.grid_search_job.GridSearchExperiment(config)
  # elif config['job']['type'] == 'bayesian':
  #   job.bayesian_optimization_job.BayesianOptimizationExperiment(config)
  # else:
  #   raise ValueError('Unknown experiment type')

  # Run/evaluate job
