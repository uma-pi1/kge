#!/usr/bin/env python
import datetime
import argparse
import os
import yaml

import kge
from kge import Dataset
from kge import Config
from kge.job import Job
from kge.util.misc import get_git_revision_short_hash, filename_in_module


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # default config
    config = Config()

    # define short option names
    short_options = {'dataset.name': '-d',
                     'job.type': '-j',
                     'train.max_epochs': '-e',
                     'model': '-m',
                     'output.folder': '-o'}

    # create parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    for key, value in Config.flatten(config.options).items():
        short = short_options.get(key)
        argtype = type(value)
        if argtype == bool:
            argtype = argparse_bool_type
        if short:
            parser.add_argument('--'+key, short, type=argtype)
        else:
            parser.add_argument('--'+key, type=argtype)
    args = parser.parse_args()

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
        config.load(args.config)

    # optionally: load configuration of resumed job
    if args.resume is not None:
        configfile = args.resume
        if os.path.isdir(configfile) \
           and os.path.isfile(configfile + '/config.yaml'):
            configfile += '/config.yaml'
        print('Resuming from configuration {}...'.format(configfile))
        config.load(configfile)
        if config.folder() == '' or not os.path.exists(config.folder()):
            raise ValueError("{} is not a valid config file for resuming"
                             .format(args.resume))

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in ['config', 'resume']:
            continue
        if value is not None:
            config.set(key, value)

    # TODO For now, relative directories (output folder, dataset folder) are
    # hard-coded to refer to # the kge base folder. This is really not a nice
    # solution, but will do for now.
    os.chdir(filename_in_module(kge, '../'))

    # initialize output folder
    if config.folder() == '':  # means: set default
        config.set('output.folder',
                   'local/experiments/'
                   + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                   + "-" + config.get('dataset.name')
                   + "-" + config.get('model'))
    if not args.resume and not config.init_folder():
        raise ValueError("output folder exists")

    # log configuration
    config.log("Configuration:")
    config.log(yaml.dump(config.options), prefix='  ')
    config.log('git commit: {}'.format(get_git_revision_short_hash()),
               prefix='  ')

    # load data
    dataset = Dataset.load(config)

    # let's go
    job = Job.create(config, dataset)
    if args.resume:
        job.resume()
    job.run()
