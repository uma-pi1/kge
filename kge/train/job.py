import torch
from kge.model import KgeModel, ComplEx


class TrainingJob:
    """Job to train a single model with a fixed set of hyperparameters. Used by
experiments such as grid search or Bayesian optimization."""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = config
        self.model = KgeModel.create(config, dataset)

    def create(config, dataset):
        """Factory method to create a training job and add necessary indexes to the
dataset (if not present)."""
        if config.get('train.type') == '1toN':
            return TrainingJob1toN(config, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run(self):
        self.config.log('Starting training...')
        # for n in range(self.config.get('train.max_epochs')):
        #     self.config.log('Starting epoch {}...'.format(n))
        #     self.epoch()

    # TODO methods for checkpointing, logging, ...


class TrainingJob1toN(TrainingJob):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        config.log("Initializing 1-to-N training job...")

        # create sp and po indexes (if not done before)
        self.train_sp = dataset.indexes.get('train_sp')
        if not self.train_sp:
            self.train_sp = TrainingJob1toN._index(dataset.train[:, [0, 1]], dataset.train[:, 2])
            config.log("{} distinct sp pairs in train".format(len(self.train_sp)),
                                 prefix='  ')
            dataset.indexes['train_sp'] = self.train_sp
        self.train_po = dataset.indexes.get('train_po')
        if not self.train_po:
            self.train_po = TrainingJob1toN._index(dataset.train[:, [1, 2]], dataset.train[:, 0])
            config.log("{} distinct po pairs in train".format(len(self.train_po)),
                                 prefix='  ')
            dataset.indexes['train_po'] = self.train_po

            # TODO index dataset
            # create optimizers, losses, ... (partly in super?)

    def _index(key, value):
        result = {}
        for i in range(len(key)):
            k = (key[i, 0].item(), key[i, 1].item())
            values = result.get(k)
            if values is None:
                values = []
                result[k] = values
            values.append(value[i])
        for key in result:
            result[key] = torch.IntTensor(sorted(result[key]))
        return result

    def epoch(self):
        pass
