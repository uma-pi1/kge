import os
import functools
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
        self.valid_job = EvaluationJob.create(config.clone(), dataset, self.model, 'valid')
        self.valid_job.config.set('eval.trace_examples', False)
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

            # validate
            if self.config.get('valid.every') > 0 \
               and self.epoch % self.config.get('valid.every') == 0:
                self.valid_job.epoch = self.epoch
                metrics = self.valid_job.run()

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
        train_sp = dataset.index_1toN('train', 'sp')
        train_po = dataset.index_1toN('train', 'po')

        # convert indexes to pytoch tensors: a nx2 keys tensor (rows = keys),
        # an offset vector (row = starting offset in values for corresponding
        # key), a values vector (entries correspond to values of original
        # index)
        #
        # Afterwards, it holds:
        # index[keys[i]] = values[offsets[i]:offsets[i+1]]
        def prepare_index(index):
            keys = torch.tensor(list(index.keys()), dtype=torch.int)
            values = torch.cat(list(index.values()))
            offsets = torch.cumsum(
                torch.tensor([0] + list(map(len, index.values())),
                             dtype=torch.int), 0)
            return keys, values, offsets
        self.train_sp_keys, self.train_sp_values, self.train_sp_offsets = \
            prepare_index(train_sp)
        self.train_po_keys, self.train_po_values, self.train_po_offsets = \
            prepare_index(train_po)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(len(train_sp)+len(train_po)),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=config.get('train.num_workers'),
            pin_memory=config.get('train.pin_memory')
           )

        self.num_examples = len(train_sp)+len(train_po)

        # TODO currently assuming BCE loss
        self.config.check('train.loss', ['bce'])

    def _get_collate_fun(self):
        num_sp = len(self.train_sp_keys)
        num_po = len(self.train_po_keys)

        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a triple of:

            - pairs (nx2 tensor, row = sp or po indexes),
            - label coordinates (position of ones in a batch_size x num_entities tensor)
            - is_sp (vector of size n, 1 if corresponding example_index is sp, 0 if po)

            """
            # count how many labels we have
            num_ones = 0
            for example_index in batch:
                if example_index < num_sp:
                    num_ones += self.train_sp_offsets[example_index+1]
                    num_ones -= self.train_sp_offsets[example_index]
                else:
                    example_index -= num_sp
                    num_ones += self.train_po_offsets[example_index+1]
                    num_ones -= self.train_po_offsets[example_index]

            # now create the results
            pairs = torch.zeros([len(batch), 2], dtype=torch.long)
            is_sp = torch.zeros([len(batch)], dtype=torch.long)
            label_coords = torch.zeros([num_ones, 2], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                is_sp[batch_index] = 1 if example_index < num_sp else 0
                if is_sp[batch_index]:
                    keys = self.train_sp_keys
                    offsets = self.train_sp_offsets
                    values = self.train_sp_values
                else:
                    example_index -= num_sp
                    keys = self.train_po_keys
                    offsets = self.train_po_offsets
                    values = self.train_po_values

                pairs[batch_index, ] = keys[example_index]
                start = offsets[example_index]
                end = offsets[example_index+1]
                size = end-start
                label_coords[current_index:(current_index+size), 0] = \
                    batch_index
                label_coords[current_index:(current_index+size), 1] = \
                    values[start:end]
                current_index += size

            # all done
            return pairs, label_coords, is_sp

        return collate

    def run_epoch(self):
        # TODO refactor: much of this can go to TrainingJob
        sum_loss = 0
        epoch_time = -time.time()
        prepare_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        for batch_index, batch in enumerate(self.loader):
            batch_prepare_time = -time.time()
            pairs = batch[0].to(self.device)
            batch_size = len(pairs)
            label_coords = batch[1].to(self.device)
            is_sp = batch[2]
            sp_indexes = is_sp.nonzero().to(self.device).view(-1)
            po_indexes = (is_sp == 0).nonzero().to(self.device).view(-1)
            labels = kge.job.util.coord_to_sparse_tensor(
                batch_size, self.dataset.num_entities, label_coords,
                self.device).to_dense()
            batch_prepare_time += time.time()
            prepare_time += batch_prepare_time

            # forward pass
            batch_forward_time = -time.time()
            scores_sp = self.model.score_sp(pairs[sp_indexes, 0],
                                            pairs[sp_indexes, 1])
            loss_value = self.loss(scores_sp.view(-1),
                                   labels[sp_indexes, ].view(-1))
            scores_po = self.model.score_po(pairs[po_indexes, 0],
                                            pairs[po_indexes, 1])
            loss_value += self.loss(scores_po.view(-1),
                                    labels[po_indexes, ].view(-1))
            loss_value = loss_value
            sum_loss += loss_value.item()*batch_size
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
                job='train', type='1toN', scope='batch',
                epoch=self.epoch,
                batch=batch_index, size=batch_size, batches=len(self.loader),
                avg_loss=loss_value.item(),
                prepare_time=batch_prepare_time,
                forward_time=batch_forward_time,
                backward_time=batch_backward_time,
                optimizer_time=batch_optimizer_time
            )
            print('\033[K\r', end="")  # clear line and go back
            print(('  batch:{: '
                   + str(1+int(math.ceil(math.log10(len(self.loader)))))
                   + 'd}/{}, avg_loss: {:14.4f}, time: {:8.4f}s').format(
                       batch_index, len(self.loader)-1, loss_value.item()/batch_size,
                       batch_prepare_time + batch_forward_time + batch_backward_time
                   + batch_optimizer_time), end='')

        epoch_time += time.time()
        print("\033[2K\r", end="")  # clear line and go back
        other_time = epoch_time - prepare_time - forward_time \
            - backward_time - optimizer_time
        self.config.trace(
            echo=True, echo_prefix="  ", log=True,
            job='train', type='1toN', scope='epoch',
            epoch=self.epoch, batches=len(self.loader),
            size=self.num_examples,
            avg_loss=sum_loss/self.num_examples,
            epoch_time=epoch_time, prepare_time=prepare_time,
            forward_time=forward_time, backward_time=backward_time,
            optimizer_time=optimizer_time,
            other_time=other_time)
