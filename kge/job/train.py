import os
import math
import torch
import torch.utils.data
import time
from kge.job import Job
from kge.model import KgeModel
from kge.util import KgeLoss
from kge.util import KgeOptimizer
import kge.job.util


class TrainingJob(Job):
    """Job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as grid search (:class:`GridJob`) or Bayesian
    optimization.

    """

    def __init__(self, config, dataset):
        from kge.job import EvaluationJob

        super().__init__(config, dataset)
        self.model = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.loss = KgeLoss.create(config)
        self.batch_size = config.get('train.batch_size')
        self.device = self.config.get('job.device')
        self.evaluation = EvaluationJob.create(config, dataset, self.model)
        self.epoch = 0
        self.model.train()

    def create(config, dataset):
        """Factory method to create a training job and add necessary label_coords to the
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
            self.save(self.config.checkpointfile(self.epoch))
            if self.epoch > 1:
                if not (checkpoint > 0 and ((self.epoch-1) % checkpoint == 0)):
                    self.config.log('Removing old checkpoint {}...'.format(
                        self.config.checkpointfile(self.epoch-1)))
                    os.remove(self.config.checkpointfile(self.epoch-1))

            # evaluate
            # if not self.epoch % self.config.get('valid.every'):
            #     self.evaluation.run()
        self.config.log('Maximum number of epochs reached.')

    def save(self, filename):
        self.config.log('Saving checkpoint to "{}"...'.format(filename))
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename)

    def load(self, filename):
        self.config.log('Loading checkpoint from "{}"...'.format(filename))
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.model.train()

    def resume(self):
        checkpointfile = self.config.last_checkpointfile()
        if checkpointfile is not None:
            self.load(checkpointfile)
        else:
            self.config.log("No checkpoint found, starting from scratch...")


class TrainingJob1toN(TrainingJob):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        config.log("Initializing 1-to-N training job...")

        # create sp and po label_coords (if not done before)
        self.train_sp = dataset.index_1toN('train', 'sp')
        self.train_po = dataset.index_1toN('train', 'po')

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            dataset.train,
            collate_fn=self._collate,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=config.get('train.num_workers'),
            pin_memory=config.get('train.pin_memory')
        )

        # TODO currently assuming BCE loss
        self.config.check('train.loss', ['bce'])

    def _collate(self, batch):
        """Return batch and label coordinates (position of ones in a batch_size x
2num_entities tensor)

        """
        label_coords = kge.job.util.get_batch_sp_po_coords(
            batch, self.dataset.num_entities, self.train_sp, self.train_po)
        batch = torch.cat(batch).reshape((-1, 3))
        return batch, label_coords

    def run_epoch(self):
        # TODO refactor: much of this can go to TrainingJob
        sum_loss = 0
        epoch_time = -time.time()
        prepare_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        for i, batch_coords in enumerate(self.loader):
            batch_prepare_time = -time.time()
            batch = batch_coords[0].to(self.device)
            label_coords = batch_coords[1].to(self.device)
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2*self.dataset.num_entities, label_coords,
                self.device)
            batch_prepare_time += time.time()
            prepare_time += batch_prepare_time

            # forward pass
            batch_forward_time = -time.time()
            scores = self.model.score_sp_po(
                batch[:, 0], batch[:, 1], batch[:, 2])
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
                epoch=self.epoch,
                batch=i, size=len(batch), batches=len(self.loader),
                avg_loss=loss_value.item()/len(batch),
                prepare_time=batch_prepare_time,
                forward_time=batch_forward_time,
                backward_time=batch_backward_time,
                optimizer_time=batch_optimizer_time
            )
            print('\033[K\r', end="")  # clear line and go back
            print(('  batch:{: '
                   + str(1+int(math.ceil(math.log10(len(self.loader)))))
                   + 'd}/{}, avg_loss: {:14.4f}, time: {:8.4f}s').format(
                i, len(self.loader)-1, loss_value.item()/len(batch),
                batch_prepare_time + batch_forward_time + batch_backward_time
                + batch_optimizer_time), end='')

        epoch_time += time.time()
        print("\033[2K\r", end="")  # clear line and go back
        other_time = epoch_time - prepare_time - forward_time \
            - backward_time - optimizer_time
        self.config.trace(
            echo=True, echo_prefix="  ", log=True,
            type='epoch', epoch=self.epoch, batches=len(self.loader),
            size=len(self.dataset.train),
            avg_loss=sum_loss/len(self.dataset.train),
            epoch_time=epoch_time, prepare_time=prepare_time,
            forward_time=forward_time, backward_time=backward_time,
            optimizer_time=optimizer_time,
            other_time=other_time)
