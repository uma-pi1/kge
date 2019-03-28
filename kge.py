import datetime
import yaml
import os
import glob
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
                     'train.max_epochs':'-e',
                     'model.type':'-m',
                     'output.folder':'-o'}

    # parse arguments
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

    # load user config file (overwrites defaults)
    if args.config is not None:
        if args.resume is not None:
            raise ValueError("config and resume")
        print('Loading configuration {}...'.format(args.config))
        config.load(args.config)

    # resume previous model
    if args.resume is not None:
        print('Resuming from configuration {}...'.format(args.resume))
        config.load(args.resume)
        if config.folder() == '' or not os.path.exists(config.folder()):
            raise ValueError("{} is not a valid config file for resuming".format(args.resume))

    # overwrite with command line arguments
    for key, value in vars(args).items():
        if key=='config' or key=='resume':
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
        if args.resume is None:
            raise NotImplementedError("output folder exists")
    else:
        os.makedirs(config.folder())

    # store full configuration in output folder
    if args.resume is None:
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
        if args.resume:
            # find last checkpoint file (stupid but works)
            tried_epoch = 0
            found_epoch = 0
            while tried_epoch < found_epoch + 100:
                tried_epoch += 1
                if os.path.exists(config.checkpointfile(tried_epoch)):
                    found_epoch = tried_epoch
            if found_epoch>0:
                job.resume(config.checkpointfile(found_epoch))
            else:
                config.log("No checkpoint found, starting from scratch...")
        job.run()
    else:
        raise ValueError("unknown job type")
