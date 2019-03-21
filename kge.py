import datetime
import yaml
import os
import argparse
from kge import job
from kge import Config

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

    # TODO load data

    # let's go
    if config.raw['experiment']['type'] == 'train':
        ## train model with specified hyperparmeters
        ## TODO create job
        pass
    else:
        raise NotImplementedError("experiment")
