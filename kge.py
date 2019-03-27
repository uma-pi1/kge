import datetime
import yaml
import os
import argparse
from kge.data import Dataset
from kge import Config
from kge.train import TrainingJob

if __name__ == '__main__':
    # default config
    config = Config()

    # define short option names
    short_options = {'dataset.name':'-d',
                     'job.type':'-j',
                     'job.device':'-r',
                     'model.type':'-m',
                     'output.folder':'-o'}

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    for key, value in Config.flatten(config.options).items():
        short = short_options.get(key)
        if short:
            parser.add_argument('--'+key, short, type=type(value))
        else:
            parser.add_argument('--'+key, type=type(value))
    args = parser.parse_args()

    # load user config file (overwrites defaults)
    if args.config is not None:
        print('Loading configuration {}...'.format(args.config))
        config.load(args.config)

    # overwrite with command line arguments
    for key, value in vars(args).items():
        if key=='config':
            continue
        if value is not None:
            config.set(key, value)

    # validate arguments and set defaults
    if config.folder() == '':
        config.set('output.folder', \
            'local/experiments/' \
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") \
            + "-" + config.get('dataset.name') \
            + "-" + config.get('model.type'))

    # create output folder
    if os.path.exists(config.folder()):
        # TODO
        raise NotImplementedError("resume/overwrite")
    else:
        os.makedirs(config.folder())

    # store full configuration in output folder
    config.dump(config.folder() + "/config.yaml")
    config.log("Configuration:")

    # also show on screen (perhaps: non-default options only?)
    config.log( yaml.dump(config.options), prefix='  ')

    # load data
    dataset = Dataset.load(config)

    # let's go
    if config.get('job.type') == 'train':
        ## train model with specified hyperparmeters
        ## TODO create job
        job = TrainingJob.create(config, dataset)
        job.run()
    else:
        raise ValueError("unknown job type")
