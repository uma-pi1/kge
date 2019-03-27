import torch
import torch.utils.data
import time
from kge.model import KgeModel
from kge.util import KgeLoss
from kge.util import KgeOptimizer


class TrainingJob:
    """Job to train a single model with a fixed set of hyperparameters. Used by
experiments such as grid search or Bayesian optimization."""

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = config
        self.model = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.loss = KgeLoss.create(config)

    def create(config, dataset):
        """Factory method to create a training job and add necessary indexes to the
dataset (if not present)."""
        if config.get('train.type') == '1toN':
            return TrainingJob1toN(config, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def epoch(self, current_epoch):
        raise NotImplementedError

    def run(self):
        self.config.log('Starting training...')
        for n in range(self.config.get('train.max_epochs')):
            self.config.log('Starting epoch {}...'.format(n))
            self.epoch(n)

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

        # create dataloader
        self.loader = torch.utils.data.DataLoader(dataset.train,
                                                  # collate_fn=self._collate,
                                                  shuffle=True,
                                                  batch_size=config.get('train.batch_size'),
                                                  num_workers=config.get('train.num_workers'),
                                                  pin_memory=config.get('train.pin_memory'))

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

    def _collate(self, batch):
        """ Produces batch as well as corresponding label matrix """

    # TODO devices
    def epoch(self, current_epoch):
        sum_loss = 0
        epoch_time = -time.time()
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        for i, batch in enumerate(self.loader):
            # forward pass
            forward_time -= time.time()
            scores = self.model.score_sp_and_po(batch[:, 0], batch[:, 1], batch[:, 2], is_training=True)
            loss = self.loss(scores, labels)
            sum_loss += loss.item()
            forward_time += time.time()

            # backward pass
            backward_time -= time.time()
            self.optimizer.zero_grad()
            self.loss.backward()
            backward_time += time.time()

            # upgrades
            optimizer_time -= time.time()
            self.optimizer.step()
            optimizer_time += time.time()
        epoch_time += time.time()

        print("epoch={} avg_loss={:.2f} forward={:.3f}s backward={:.3f}s opt={:.3f}s other={:.3f}s total={:.3f}s".format(
            current_epoch, sum_loss / i, forward_time, backward_time, optimizer_time,
            epoch_time - forward_time - backward_time - optimizer_time, epoch_time))





