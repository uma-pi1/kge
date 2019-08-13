import itertools
import os
import math
import time
import torch
import torch.utils.data

from kge import Dataset
from kge.job import Job
from kge.model import KgeModel
from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
import kge.job.util


class TrainingJob(Job):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_compute_batch_loss`.

    """

    def __init__(self, config, dataset, parent_job=None):
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        self.model = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.lr_scheduler, self.metric_based_scheduler = KgeLRScheduler.create(
            config, self.optimizer
        )
        self.loss = KgeLoss.create(config)
        self.batch_size = config.get("train.batch_size")
        self.device = self.config.get("job.device")
        valid_conf = config.clone()
        valid_conf.set("job.type", "eval")
        valid_conf.set("eval.data", "valid")
        valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
        self.valid_job = EvaluationJob.create(
            valid_conf, dataset, parent_job=self, model=self.model
        )
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch = self.config.get("train.trace_level") == "batch"
        self.epoch = 0
        self.valid_trace = []
        self.is_prepared = False
        self.model.train()

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str = None

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks = []

        #: Hooks run after a validation job.
        #: Signature: job, trace_entry
        self.post_valid_hooks = []

    @staticmethod
    def create(config, dataset, parent_job=None):
        """Factory method to create a training job and add necessary label_coords to
the dataset (if not present).

        """
        if config.get("train.type") == "1toN":
            return TrainingJob1toN(config, dataset, parent_job)
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(config, dataset, parent_job)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run(self):
        """Start/resume the training job and run to completion."""
        self.config.log("Starting training...")
        checkpoint_every = self.config.get("checkpoint.every")
        checkpoint_keep = self.config.get("checkpoint.keep")
        metric_name = self.config.get("valid.metric")
        patience = self.config.get("valid.early_stopping.patience")
        while True:
            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break

            # checking for model improvement according to metric_name
            # and do early stopping and keep the best checkpoint
            if len(self.valid_trace) > 0:
                best_index = max(
                    range(len(self.valid_trace)),
                    key=lambda index: self.valid_trace[index][metric_name],
                )
                if best_index == len(self.valid_trace) - 1:
                    self.save(self.config.checkpoint_file("best"))
                if (
                    patience > 0
                    and len(self.valid_trace) > patience
                    and best_index < len(self.valid_trace) - patience
                ):
                    self.config.log(
                        "Stopping early ({} did not improve over best result "
                        + "in the last {} validation runs).".format(
                            metric_name, patience
                        )
                    )
                    break
                if self.epoch > self.config.get(
                    "valid.early_stopping.min_threshold.epochs"
                ) and self.valid_trace[best_index][metric_name] < self.config.get(
                    "valid.early_stopping.min_threshold.metric_value"
                ):
                    self.config.log(
                        "Stopping early ({} did not achieve min treshold after {} epochs".format(
                            metric_name, self.epoch
                        )
                    )
                    break

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            for f in self.post_epoch_hooks:
                f(self, trace_entry)
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            # validate
            if (
                self.config.get("valid.every") > 0
                and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self, trace_entry)
                self.model.meta["valid_trace_entry"] = trace_entry

                # metric-based scheduler step
                if self.metric_based_scheduler:
                    self.lr_scheduler.step(trace_entry[metric_name])

            # epoch-based scheduler step
            if self.lr_scheduler and not self.metric_based_scheduler:
                self.lr_scheduler.step(self.epoch)

            # create checkpoint and delete old one, if necessary
            self.save(self.config.checkpoint_file(self.epoch))
            if self.epoch > 1:
                delete_checkpoint_epoch = -1
                if checkpoint_every == 0:
                    # do not keep any old checkpoints
                    delete_checkpoint_epoch = self.epoch - 1
                elif (self.epoch - 1) % checkpoint_every != 0:
                    # delete checkpoints that are not in the checkpoint.every schedule
                    delete_checkpoint_epoch = self.epoch - 1
                elif checkpoint_keep > 0:
                    # keep a maximum number of checkpoint_keep checkpoints
                    delete_checkpoint_epoch = (
                        self.epoch - 1 - checkpoint_every * checkpoint_keep
                    )
                if delete_checkpoint_epoch > 0:
                    self.config.log(
                        "Removing old checkpoint {}...".format(
                            self.config.checkpoint_file(delete_checkpoint_epoch)
                        )
                    )
                    os.remove(self.config.checkpoint_file(delete_checkpoint_epoch))

    def save(self, filename):
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "config": self.config,
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model": self.model.save(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "job_id": self.job_id,
            },
            filename,
        )

    def load(self, filename):
        """Load job state from specified file.

        Returns job id of the job that created the checkpoint."""
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename)
        if "model" in checkpoint:
            # new format
            self.model.load(checkpoint["model"])
        else:
            # old format (deprecated, will eventually be removed)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()
        return checkpoint.get("job_id")

    def resume(self):
        """Load job state from last checkpoint.

        Job is not actually run using this method; follow up with `run` for this."""
        last_checkpoint = self.config.last_checkpoint()
        if last_checkpoint is not None:
            checkpoint_file = self.config.checkpoint_file(last_checkpoint)
            self.resumed_from_job = self.load(checkpoint_file)
            self.config.log("Resumed from job {}".format(self.resumed_from_job))
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    def run_epoch(self) -> dict:
        "Runs an epoch and returns a trace entry."

        # prepare the job is not done already
        if not self.is_prepared:
            self._prepare()
            self.model.prepare_job(self)  # let the model add some hooks
            self.is_prepared = True

        # variables that record various statitics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = []
        epoch_time = -time.time()
        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0

        # process each batch
        for batch_index, batch in enumerate(self.loader):
            for f in self.pre_batch_hooks:
                f(self)

            # preprocess batch and perform forward pass
            loss_value, batch_size, batch_prepare_time, batch_forward_time = self._compute_batch_loss(
                batch_index, batch
            )
            sum_loss += loss_value.item() * batch_size
            prepare_time += batch_prepare_time

            # determine penalty terms (part of forward pass)
            batch_forward_time -= time.time()
            penalty_value = torch.zeros(1, device=self.device)
            penalty_values = self.model.penalty(
                epoch=self.epoch, batch_index=batch_index, num_batches=len(self.loader)
            )
            for pv_index, pv_value in enumerate(penalty_values):
                penalty_value = penalty_value + pv_value
                if len(sum_penalties) > pv_index:
                    sum_penalties[pv_index] += pv_value.item()
                else:
                    sum_penalties.append(pv_value.item())
            sum_penalty += penalty_value.item()
            batch_forward_time += time.time()
            forward_time += batch_forward_time

            # backward pass
            batch_backward_time = -time.time()
            cost_value = loss_value + penalty_value
            cost_value.backward()
            batch_backward_time += time.time()
            backward_time += batch_backward_time

            # update parameters
            batch_optimizer_time = -time.time()
            self.optimizer.step()
            batch_optimizer_time += time.time()
            optimizer_time += batch_optimizer_time

            # tracing/logging
            if self.trace_batch:
                batch_trace = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "batch": batch_index,
                    "size": batch_size,
                    "batches": len(self.loader),
                    "avg_loss": loss_value.item(),
                    "penalties": [p.item() for p in penalty_values],
                    "penalty": penalty_value.item(),
                    "cost": cost_value.item(),
                    "prepare_time": batch_prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                }
                for f in self.post_batch_trace_hooks:
                    f(self, batch_trace)
                self.trace(**batch_trace)
            print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    loss_value.item(),
                    penalty_value.item(),
                    cost_value.item(),
                    batch_prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

        # all done; now trace and log
        epoch_time += time.time()
        print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )
        trace_entry = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            batches=len(self.loader),
            size=self.num_examples,
            avg_loss=sum_loss / self.num_examples,
            avg_penalty=sum_penalty / len(self.loader),
            avg_penalties=[p / len(self.loader) for p in sum_penalties],
            avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
            epoch_time=epoch_time,
            prepare_time=prepare_time,
            forward_time=forward_time,
            backward_time=backward_time,
            optimizer_time=optimizer_time,
            other_time=other_time,
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)
        return trace_entry

    def _prepare(self):
        """Prepare this job for running.

        Sets (at least) the `loader`, `num_examples`, and `type_str` attributes of this
        job to a data loader, number of examples per epoch, and a name for the trainer,
        repectively.

        Guaranteed to be called exactly once before running the first epoch.

        """
        raise NotImplementedError

    def _compute_batch_loss(self, batch_index, batch):
        "Returns loss_value (avg over batch), batch size, prepare time, forward time."
        raise NotImplementedError


class TrainingJob1toN(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.label_smoothing = config.check_range(
            "1toN.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception("Label_smoothing was set to {}, "
                                "should be at least 0.".format(self.label_smoothing))
        elif (self.label_smoothing > 0 and
              self.label_smoothing <= (1.0 / dataset.num_entities)):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/dataset.num_entities = {}, "
                    "was set to {}.".format(1.0 / dataset.num_entities, self.label_smoothing)
                )
                self.label_smoothing = 1.0 / dataset.num_entities
            else:
                raise Exception("Label_smoothing was set to {}, "
                                "should be at least {}.".format(self.label_smoothing,
                                                                1.0 / dataset.num_entities))

        config.log("Initializing 1-to-N training job...")

    def _prepare(self):
        self.type_str = "1toN"

        # create sp and po label_coords (if not done before)
        train_sp = self.dataset.index_1toN("train", "sp")
        train_po = self.dataset.index_1toN("train", "po")

        # convert indexes to pytoch tensors: a nx2 keys tensor (rows = keys),
        # an offset vector (row = starting offset in values for corresponding
        # key), a values vector (entries correspond to values of original
        # index)
        #
        # Afterwards, it holds:
        # index[keys[i]] = values[offsets[i]:offsets[i+1]]

        self.train_sp_keys, self.train_sp_values, self.train_sp_offsets = \
            Dataset.prepare_index(train_sp)
        self.train_po_keys, self.train_po_values, self.train_po_offsets = \
            Dataset.prepare_index(train_po)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(len(train_sp) + len(train_po)),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )
        self.num_examples = len(train_sp) + len(train_po)

    def _get_collate_fun(self):
        num_sp = len(self.train_sp_keys)

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
                    num_ones += self.train_sp_offsets[example_index + 1]
                    num_ones -= self.train_sp_offsets[example_index]
                else:
                    example_index -= num_sp
                    num_ones += self.train_po_offsets[example_index + 1]
                    num_ones -= self.train_po_offsets[example_index]

            # now create the results
            sp_po_batch = torch.zeros([len(batch), 2], dtype=torch.long)
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

                sp_po_batch[batch_index,] = keys[example_index]
                start = offsets[example_index]
                end = offsets[example_index + 1]
                size = end - start
                label_coords[current_index : (current_index + size), 0] = batch_index
                label_coords[current_index : (current_index + size), 1] = values[
                    start:end
                ]
                current_index += size

            # all done
            return sp_po_batch, label_coords, is_sp

        return collate

    def _compute_batch_loss(self, batch_index, batch):
        # prepare
        batch_prepare_time = -time.time()
        sp_po_batch = batch[0].to(self.device)
        batch_size = len(sp_po_batch)
        label_coords = batch[1].to(self.device)
        is_sp = batch[2]
        sp_indexes = is_sp.nonzero().to(self.device).view(-1)
        po_indexes = (is_sp == 0).nonzero().to(self.device).view(-1)
        labels = kge.job.util.coord_to_sparse_tensor(
            batch_size, self.dataset.num_entities, label_coords, self.device
        ).to_dense()
        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            labels = (1.0 - self.label_smoothing) * labels + 1.0 / labels.size(1)
        batch_prepare_time += time.time()

        # forward pass
        batch_forward_time = -time.time()
        self.optimizer.zero_grad()
        loss_value = torch.zeros(1, device=self.device)
        if len(sp_indexes) > 0:
            scores_sp = self.model.score_sp(sp_po_batch[sp_indexes, 0], sp_po_batch[sp_indexes, 1])
            loss_value = loss_value + self.loss(
                scores_sp.view(-1), labels[sp_indexes,].view(-1)
            )
        if len(po_indexes) > 0:
            scores_po = self.model.score_po(sp_po_batch[po_indexes, 0], sp_po_batch[po_indexes, 1])
            loss_value = loss_value + self.loss(
                scores_po.view(-1), labels[po_indexes,].view(-1)
            )
        batch_forward_time += time.time()

        return loss_value, batch_size, batch_prepare_time, batch_forward_time


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self._sampler = KgeSampler.create(config, dataset)
        config.log("Initializing negative sampling training job...")
        self.is_prepared = False

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.loader = torch.utils.data.DataLoader(
            range(self.dataset.train.size(0)),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )
        self.num_examples = self.dataset.train.size(0)

        self.is_prepared = True

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n * num_negatives, 3], row = sp or po indexes),
            - labels (tensor of [n * num_negatives] with labels for triples)

            """
            triples = []
            labels = []
            for batch_index, example_index in enumerate(batch):
                triples.append(self.dataset.train[example_index].type(torch.long))
                labels.extend([1] + [0] * (self._sampler.num_negatives))

            triples = (
                torch.stack(triples)
                .repeat(1, 1 + self._sampler.num_negatives)
                .view(-1, 3)
            )

            offset = 0
            for slot, slot_num_negatives, voc_size in [
                ([0], self._sampler._num_negatives_s, self.dataset.num_entities),
                ([1], self._sampler._num_negatives_p, self.dataset.num_relations),
                ([2], self._sampler._num_negatives_o, self.dataset.num_entities),
            ]:
                triples[
                    list(
                        itertools.chain(
                            *map(
                                lambda x: range(x + 1, x + slot_num_negatives + 1),
                                range(
                                    offset,
                                    triples.size(0),
                                    1 + self._sampler.num_negatives,
                                ),
                            )
                        )
                    ),
                    (slot * slot_num_negatives) * len(batch),
                ] = self._sampler.sample(voc_size, slot_num_negatives * len(batch))
                offset += slot_num_negatives
            return triples, torch.tensor(labels, dtype=torch.float)

        return collate

    def _compute_batch_loss(self, batch_index, batch):
        # prepare
        batch_prepare_time = -time.time()
        triples = batch[0].to(self.device)
        batch_size = len(triples)
        labels = batch[1].to(self.device)
        batch_prepare_time += time.time()

        # forward pass
        batch_forward_time = -time.time()
        self.optimizer.zero_grad()
        loss_value = torch.zeros(1, device=self.device)
        scores = self.model.score_spo(triples[:, 0], triples[:, 1], triples[:, 2])
        loss_value = loss_value + self.loss(scores, labels)
        batch_forward_time += time.time()

        return loss_value, batch_size, batch_prepare_time, batch_forward_time
