import os
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
        self.dataset = dataset
        self.model = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.loss = KgeLoss.create(config)
        self.batch_size = config.get('train.batch_size')
        self.device = self.config.get('job.device')
        self.epoch = 0
        self.model.train()

    def create(config, dataset):
        """Factory method to create a training job and add necessary indexes to the
dataset (if not present)."""
        if config.get('train.type') == '1toN':
            return TrainingJob1toN(config, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run_epoch(self):
        raise NotImplementedError

    def run(self):
        self.config.log('Starting training...')
        checkpoint = self.config.get('checkpoint.every')
        while self.epoch < self.config.get('train.max_epochs'):
            self.epoch += 1
            self.config.log('Starting epoch {}...'.format(self.epoch))
            self.run_epoch()
            self.config.log('Finished epoch {}.'.format(self.epoch))

            # create checkpoint and delete old one, if necessary
            self.checkpoint(self.config.checkpointfile(self.epoch))
            if self.epoch>1:
                if not (checkpoint > 0 and ((self.epoch-1) % checkpoint == 0)):
                    self.config.log('Removing old checkpoint {}...'.format(
                        self.config.checkpointfile(self.epoch-1)))
                    os.remove(self.config.checkpointfile(self.epoch-1))
        self.config.log('Maximum number of epochs reached.')

    def checkpoint(self, filename):
        self.config.log('Checkpointing to "{}"...'.format(filename))
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename)

    def resume(self, filename):
        self.config.log('Loading checkpoint from "{}"...'.format(filename))
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.model.train()

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
                                                  collate_fn=self._collate,
                                                  shuffle=True,
                                                  batch_size=self.batch_size,
                                                  num_workers=config.get('train.num_workers'),
                                                  pin_memory=config.get('train.pin_memory'))

        # TODO currently assuming BCE loss
        self.config.check('train.loss', [ 'bce' ])

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
            result[key] = torch.LongTensor(sorted(result[key]))
        return result

    def _collate(self, batch):
        "Returns batch and label indexes (position of ones in a batch_size x 2num_entities tensor)"
        n_E = self.dataset.num_entities
        num_indexes = 0
        for i, triple in enumerate(batch):
            s,p,o = triple[0].item(), triple[1].item(), triple[2].item()
            num_indexes += len(self.train_sp[(s,p)])
            num_indexes += len(self.train_po[(p,o)])

        indexes = torch.zeros([num_indexes, 2], dtype=torch.long)
        current_index = 0
        for i, triple in enumerate(batch):
            s,p,o = triple[0].item(), triple[1].item(), triple[2].item()

            objects = self.train_sp[(s,p)]
            indexes[current_index:(current_index+len(objects)), 0] = i
            indexes[current_index:(current_index+len(objects)), 1] = objects
            current_index += len(objects)

            subjects = self.train_po[(p,o)] + n_E
            indexes[current_index:(current_index+len(subjects)), 0] = i
            indexes[current_index:(current_index+len(subjects)), 1] = subjects
            current_index += len(subjects)
        batch = torch.cat(batch).reshape((-1,3))
        return batch, indexes

    # TODO devices
    def run_epoch(self):
        sum_loss = 0
        epoch_time = -time.time()
        prepare_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        for i, batch_labels in enumerate(self.loader):
            batch_prepare_time = -time.time()
            batch = batch_labels[0].to(self.device)
            indexes = batch_labels[1].to(self.device)
            if self.device == 'cpu':
                labels = torch.sparse.FloatTensor(
                    indexes.t(),
                    torch.ones([len(indexes)], dtype=torch.float, device=self.device),
                    torch.Size([len(batch),2*self.dataset.num_entities]))
            else:
                labels = torch.cuda.sparse.FloatTensor(
                    indexes.t(),
                    torch.ones([len(indexes)], dtype=torch.float, device=self.device),
                    torch.Size([len(batch),2*self.dataset.num_entities]),
                    device=self.device)
            batch_prepare_time += time.time()
            prepare_time += batch_prepare_time

            # forward pass
            batch_forward_time = -time.time()
            scores = self.model.score_sp_po(batch[:, 0], batch[:, 1], batch[:, 2], is_training=True)
            loss_value = self.loss(scores.view(-1), labels.to_dense().view(-1))
            sum_loss += loss_value.item()
            batch_forward_time += time.time()
            forward_time += batch_forward_time

            # backward pass
            batch_backward_time = -time.time()
            self.optimizer.zero_grad()
            loss_value.backward()
            batch_backward_time += time.time()
            backward_time += batch_backward_time

            # upgrades
            batch_optimizer_time = -time.time()
            self.optimizer.step()
            batch_optimizer_time += time.time()
            optimizer_time += batch_optimizer_time

            self.config.trace(
                type='batch',
                epoch=self.epoch, batch=i, batches=len(self.loader),
                batch_loss=loss_value.item(),
                prepare_time=batch_prepare_time,
                forward_time=batch_forward_time, backward_time=batch_backward_time,
                optimizer_time=batch_optimizer_time
            )
            print('\033[K\r', end="") # clear line and go back
            print('  batch {}/{}, loss {}, time {}'.format(
                i, len(self.loader)-1, loss_value.item(),
                batch_prepare_time+batch_forward_time+batch_backward_time
                +batch_optimizer_time), end='')

        epoch_time += time.time()
        print("\033[K\r", end="") # clear line and go back
        self.config.trace(
            echo=True, log=True,
            type='epoch', epoch=self.epoch, batches=len(self.loader),
            sum_loss = sum_loss,
            epoch_time = epoch_time, prepare_time=prepare_time,
            forward_time=forward_time, backward_time=backward_time,
            optimizer_time=optimizer_time,
            other_time=epoch_time-prepare_time-forward_time-backward_time-optimizer_time)
