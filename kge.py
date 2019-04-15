import datetime
import sys

import yaml
import os
import argparse
from kge import Dataset
from kge import Config
from kge.job import Job
from kge.util.misc import get_git_revision_short_hash

if __name__ == '__main__':
    # default config
    config = Config()

    # define short option names
    short_options = {'dataset.name': '-d',
                     'job.type': '-j',
                     'train.max_epochs': '-e',
                     'model.type': '-m',
                     'output.folder': '-o'}

    # create parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    for key, value in Config.flatten(config.options).items():
        short = short_options.get(key)
        if short:
            parser.add_argument('--'+key, short, type=type(value))
        else:
            parser.add_argument('--'+key, type=type(value))
    args = parser.parse_args()

    # load model config if specified on command line
    model_config_loaded = False
    if vars(args)['model.type'] is not None:
        config.load_model_config(vars(args)['model.type'])
        model_config_loaded = True

    # use toy config file if no config given
    if args.config is None and args.resume is None:
        args.config = 'examples/toy.yaml'
        print('WARNING: No configuration specified; using '
              + args.config)

    # optionally: load user config file (overwrites some defaults)
    if args.config is not None:
        if args.resume is not None:
            raise ValueError("config and resume")
        print('Loading configuration {}...'.format(args.config))
        config.load(args.config, detect_model_config=not model_config_loaded)

    # optionally: load configuration of resumed job
    if args.resume is not None:
        configfile = args.resume
        if os.path.isdir(configfile) \
           and os.path.isfile(configfile + '/config.yaml'):
            configfile += '/config.yaml'
        print('Resuming from configuration {}...'.format(configfile))
        config.load(configfile, detect_model_config=not model_config_loaded)
        if config.folder() == '' or not os.path.exists(config.folder()):
            raise ValueError("{} is not a valid config file for resuming"
                             .format(args.resume))

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in ['config', 'resume']:
            continue
        if value is not None:
            config.set(key, value)

    # initialize output folder
    if config.folder() == '':  # means: set default
        config.set('output.folder',
                   'local/experiments/'
                   + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                   + "-" + config.get('dataset.name')
                   + "-" + config.get('model.type'))
    if not args.resume and not config.init_folder():
        raise ValueError("output folder exists")

    # log configuration
    config.log("Configuration:")
    config.log(yaml.dump(config.options), prefix='  ')

    config.log('git commit: {}'.format(get_git_revision_short_hash()), prefix='  ')

    # load data
    dataset = Dataset.load(config)

    # let's go
    job = Job.create(config, dataset)
    if args.resume:
        job.resume()
    job.run()
