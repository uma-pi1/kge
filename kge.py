import datetime
import yaml
import os
import argparse
from kge import job
from kge.config import Config

if __name__ == '__main__':
    # default config
    config = Config()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    # TODO add all entries of config as arguments to parser
    args = parser.parse_args()

    # load user config file (overwrites defaults)
    if args.config is not None:
        config.load(args.config)

    # validate arguments and set defaults
    if config.folder() == '':
        config.raw['output']['folder'] = \
            'local/experiments/' \
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") \
            + "-" + config.raw['dataset']['name'] \
            + "-" + config.raw['model']['name']

    # create output folder
    if os.path.exists(config.folder()):
        # TODO
        raise NotImplementedError("resume")
    else:
        os.makedirs(config.folder())

    # store configuration in output folder
    config.dump(config.folder() + "/config.yaml")

    # print status information
    config.log( yaml.dump(config.raw) )

    # create job
    # if config['job']['type'] == 'grid':
    #   job.grid_search_job.GridSearchExperiment(config)
    # elif config['job']['type'] == 'bayesian':
    #   job.bayesian_optimization_job.BayesianOptimizationExperiment(config)
    # else:
    #   raise ValueError('Unknown experiment type')

    # Run/evaluate job
