import itertools
import os
import math
import time
from dataclasses import dataclass

import torch
import torch.utils.data

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from typing import Any, Callable, Dict, List, Optional
import kge.job.util

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class TrainingJob(Job):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self, config: Config, dataset: Dataset, parent_job: Job = None
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        self.model: KgeModel = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.lr_scheduler, self.metric_based_scheduler = KgeLRScheduler.create(
            config, self.optimizer
        )
        self.loss = KgeLoss.create(config)
        self.batch_size: int = config.get("train.batch_size")
        self.device: str = self.config.get("job.device")
        valid_conf = config.clone()
        valid_conf.set("job.type", "eval")
        valid_conf.set("eval.data", "valid")
        valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
        self.valid_job = EvaluationJob.create(
            valid_conf, dataset, parent_job=self, model=self.model
        )
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.valid_trace: List[Dict[str, Any]] = []
        self.is_prepared = False
        self.model.train()

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks: List[Callable[[Job, Dict[str, Any]]]] = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks: List[Callable[[Job]]] = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks: List[Callable[[Job, Dict[str, Any]]]] = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks: List[Callable[[Job, Dict[str, Any]]]] = []

        #: Hooks run after a validation job.
        #: Signature: job, trace_entry
        self.post_valid_hooks: List[Callable[[Job, Dict[str, Any]]]] = []

        #: Hooks run after training
        #: Signature: job, trace_entry
        self.post_train_hooks: List[Callable[[Job, Dict[str, Any]]]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config, dataset: Dataset, parent_job: Job = None
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "KvsAll":
            return TrainingJobKvsAll(config, dataset, parent_job)
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(config, dataset, parent_job)
        elif config.get("train.type") == "1vsAll":
            return TrainingJob1vsAll(config, dataset, parent_job)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run(self) -> None:
        """Start/resume the training job and run to completion."""
        self.config.log("Starting training...")
        checkpoint_every = self.config.get("train.checkpoint.every")
        checkpoint_keep = self.config.get("train.checkpoint.keep")
        metric_name = self.config.get("valid.metric")
        patience = self.config.get("valid.early_stopping.patience")
        while True:
            # checking for model improvement according to metric_name
            # and do early stopping and keep the best checkpoint
            if (
                len(self.valid_trace) > 0
                and self.valid_trace[-1]["epoch"] == self.epoch
            ):
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
                        "Stopping early ({} did not improve over best result ".format(
                            metric_name
                        )
                        + "in the last {} validation runs).".format(patience)
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

            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            for f in self.post_epoch_hooks:
                f(self, trace_entry)
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.module.meta["train_job_trace_entry"] = self.trace_entry
            self.model.module.meta["train_epoch"] = self.epoch
            self.model.module.meta["train_config"] = self.config
            self.model.module.meta["train_trace_entry"] = trace_entry

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
                self.model.module.meta["valid_trace_entry"] = trace_entry

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
                    if os.path.exists(
                        self.config.checkpoint_file(delete_checkpoint_epoch)
                    ):
                        self.config.log(
                            "Removing old checkpoint {}...".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )
                        os.remove(self.config.checkpoint_file(delete_checkpoint_epoch))
                    else:
                        self.config.log(
                            "Could not delete old checkpoint {}, does not exits.".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )

        for f in self.post_train_hooks:
            f(self, trace_entry)
        self.trace(event="train_completed")

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "type": "train",
                "config": self.config,
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model": self.model.module.save(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "job_id": self.job_id,
            },
            filename,
        )

    def load(self, filename: str) -> str:
        """Load job state from specified file.

        Returns job id of the job that created the checkpoint."""
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")
        if "model" in checkpoint:
            # new format
            self.model.module.load(checkpoint["model"])
        else:
            # old format (deprecated, will eventually be removed)
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.module.train()
        return checkpoint.get("job_id")

    def resume(self, checkpoint_file: str = None) -> None:
        if checkpoint_file is None:
            last_checkpoint = self.config.last_checkpoint()
            if last_checkpoint is not None:
                checkpoint_file = self.config.checkpoint_file(last_checkpoint)

        if checkpoint_file is not None:
            self.resumed_from_job_id = self.load(checkpoint_file)
            self.trace(
                event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint_file
            )
            self.config.log(
                "Resumed from {} of job {}".format(
                    checkpoint_file, self.resumed_from_job_id
                )
            )
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    def run_epoch(self) -> Dict[str, Any]:
        "Runs an epoch and returns a trace entry."

        # prepare the job is not done already
        if not self.is_prepared:
            self._prepare()
            self.model.module.prepare_job(self)  # let the model add some hooks
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

            # process batch (preprocessing + forward pass + backward pass on loss)
            self.optimizer.zero_grad()
            batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                batch_index, batch
            )
            sum_loss += batch_result.avg_loss * batch_result.size

            # determine penalty terms (forward pass)
            batch_forward_time = batch_result.forward_time - time.time()
            penalties_torch = self.model.module.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, penalty_value_torch in enumerate(penalties_torch):
                penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                if len(sum_penalties) > index:
                    sum_penalties[index] += penalty_value_torch.item()
                else:
                    sum_penalties.append(penalty_value_torch.item())
            sum_penalty += penalty
            batch_backward_time += time.time()

            # determine full cost
            cost_value = batch_result.avg_loss + penalty

            # TODO # visualize graph
            # if (
            #     self.epoch == 1
            #     and batch_index == 0
            #     and self.config.get("train.visualize_graph")
            # ):
            #     from torchviz import make_dot

            #     f = os.path.join(self.config.folder, "cost_value")
            #     graph = make_dot(cost_value, params=dict(self.model.named_parameters()))
            #     graph.save(f"{f}.gv")
            #     graph.render(f)  # needs graphviz installed
            #     self.config.log("Exported compute graph to " + f + ".{gv,pdf}")

            # print memory stats
            if self.epoch == 1 and batch_index == 0:
                if self.device.startswith("cuda"):
                    self.config.log(
                        "CUDA memory after first batch: allocated={:14,} "
                        "cached={:14,} max_allocated={:14,}".format(
                            torch.cuda.memory_allocated(self.device),
                            torch.cuda.memory_cached(self.device),
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    )

            # update parameters
            batch_optimizer_time = -time.time()
            self.model(None, None, None, combine="optim", optimizer=self.optimizer)
            #self.optimizer.step()
            batch_optimizer_time += time.time()

            # tracing/logging
            if self.trace_batch:
                batch_trace = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "batch": batch_index,
                    "size": batch_result.size,
                    "batches": len(self.loader),
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                }
                for f in self.post_batch_trace_hooks:
                    f(self, batch_trace)
                self.trace(**batch_trace, event="batch_completed")
            print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}"
                    + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    batch_result.avg_loss,
                    penalty,
                    cost_value,
                    batch_result.prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

            # update times
            prepare_time += batch_result.prepare_time
            forward_time += batch_forward_time
            backward_time += batch_backward_time
            optimizer_time += batch_optimizer_time

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
            event="epoch_completed",
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

    @dataclass
    class _ProcessBatchResult:
        """Result of running forward+backward pass on a batch."""

        avg_loss: float
        size: int
        prepare_time: float
        forward_time: float
        backward_time: float

    def _process_batch(
        self, batch_index: int, batch
    ) -> "TrainingJob._ProcessBatchResult":
        "Run forward and backward pass on batch and return results."
        raise NotImplementedError


class TrainingJobKvsAll(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.label_smoothing = config.check_range(
            "KvsAll.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "KvsAll"

        if self.__class__ == TrainingJobKvsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        # create sp and po label_coords (if not done before)
        train_sp = self.dataset.index("train_sp_to_o")
        train_po = self.dataset.index("train_po_to_s")

        # convert indexes to pytoch tensors: a nx2 keys tensor (rows = keys),
        # an offset vector (row = starting offset in values for corresponding
        # key), a values vector (entries correspond to values of original
        # index)
        #
        # Afterwards, it holds:
        # index[keys[i]] = values[offsets[i]:offsets[i+1]]

        (
            self.train_sp_keys,
            self.train_sp_values,
            self.train_sp_offsets,
        ) = kge.indexing.prepare_index(train_sp)
        (
            self.train_po_keys,
            self.train_po_values,
            self.train_po_offsets,
        ) = kge.indexing.prepare_index(train_po)

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
            - triples (all true triples in the batch: needed for weighted penalties only)

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
            label_coords = torch.zeros([num_ones, 2], dtype=torch.int)
            current_index = 0
            triples = torch.zeros([num_ones, 3], dtype=torch.long)
            for batch_index, example_index in enumerate(batch):
                is_sp[batch_index] = 1 if example_index < num_sp else 0
                if is_sp[batch_index]:
                    keys = self.train_sp_keys
                    offsets = self.train_sp_offsets
                    values = self.train_sp_values
                    sp_po_col_1, sp_po_col_2, o_s_col = S, P, O
                else:
                    example_index -= num_sp
                    keys = self.train_po_keys
                    offsets = self.train_po_offsets
                    values = self.train_po_values
                    o_s_col, sp_po_col_1, sp_po_col_2 = S, P, O

                sp_po_batch[batch_index,] = keys[example_index]
                start = offsets[example_index]
                end = offsets[example_index + 1]
                size = end - start
                label_coords[current_index : (current_index + size), 0] = batch_index
                label_coords[current_index : (current_index + size), 1] = values[
                    start:end
                ]
                triples[current_index : (current_index + size), sp_po_col_1] = keys[
                    example_index
                ][0]
                triples[current_index : (current_index + size), sp_po_col_2] = keys[
                    example_index
                ][1]
                triples[current_index : (current_index + size), o_s_col] = values[
                    start:end
                ]
                current_index += size

            # all done
            return {
                "sp_po_batch": sp_po_batch,
                "label_coords": label_coords,
                "is_sp": is_sp,
                "triples": triples,
            }

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        sp_po_batch = batch["sp_po_batch"].to(self.device)
        batch_size = len(sp_po_batch)
        label_coords = batch["label_coords"].to(self.device)
        is_sp = batch["is_sp"]
        sp_indexes = is_sp.nonzero().to(self.device).view(-1)
        po_indexes = (is_sp == 0).nonzero().to(self.device).view(-1)
        labels = kge.job.util.coord_to_sparse_tensor(
            batch_size, self.dataset.num_entities(), label_coords, self.device
        ).to_dense()
        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            labels = (1.0 - self.label_smoothing) * labels + 1.0 / labels.size(1)
        prepare_time += time.time()

        # forward/backward pass (sp)
        loss_value = 0.0
        if len(sp_indexes) > 0:
            forward_time = -time.time()
            scores_sp = self.model(sp_po_batch[sp_indexes, 0], sp_po_batch[sp_indexes, 1], None, combine='sp*')
            #scores_sp = self.model.score_sp(
            #    sp_po_batch[sp_indexes, 0], sp_po_batch[sp_indexes, 1]
            #)
            loss_value_sp = self.loss(scores_sp, labels[sp_indexes,]) / batch_size
            loss_value = +loss_value_sp.item()
            forward_time += time.time()
            backward_time = -time.time()
            loss_value_sp.backward()
            backward_time += time.time()

        # forward/backward pass (po)
        if len(po_indexes) > 0:
            forward_time = -time.time()
            scores_po = self.model(None, sp_po_batch[po_indexes, 0], sp_po_batch[po_indexes, 1], combine='*po')
            #scores_po = self.model.score_po(
            #    sp_po_batch[po_indexes, 0], sp_po_batch[po_indexes, 1]
            #)
            loss_value_po = self.loss(scores_po, labels[po_indexes,]) / batch_size
            loss_value = loss_value_po.item()
            forward_time += time.time()
            backward_time = -time.time()
            loss_value_po.backward()
            backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self.is_prepared = False
        self._implementation = self.config.get("negative_sampling.implementation")
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples.values())
            if max_nr_of_negs <= 30:
                self._implementation = "spo"
            elif max_nr_of_negs > 30:
                self._implementation = "sp_po"

        config.log(
            "Initializing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )
        self.type_str = "negative_sampling"

        if self.__class__ == TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.num_examples = self.dataset.train().size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.is_prepared = True

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.train()[batch, :].long()
            # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
            # labels[:, 0] = 1
            # labels = labels.view(-1)

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        triples = batch["triples"].to(self.device)
        negative_samples = [ns.to(self.device) for ns in batch["negative_samples"]]
        batch_size = len(triples)
        prepare_time += time.time()

        loss_value = 0.0
        forward_time = 0.0
        backward_time = 0.0
        num_samples = 0
        labels = None
        for slot in [S, P, O]:
            if self._sampler.num_samples[slot] <= 0:
                continue

            # construct gold labels: first column corresponds to positives, rest to negatives
            if self._sampler.num_samples[slot] != num_samples:
                prepare_time -= time.time()
                num_samples = self._sampler.num_samples[slot]
                labels = torch.zeros((batch_size, 1 + num_samples), device=self.device)
                labels[:, 0] = 1
                prepare_time += time.time()

            # compute corresponding scores
            scores = None
            if self._implementation == "spo":
                # construct triples
                prepare_time -= time.time()
                triples_to_score = triples.repeat(1, 1 + num_samples).view(-1, 3)
                triples_to_score[:, slot] = torch.cat(
                    (
                        triples[:, [slot]],  # positives
                        negative_samples[slot],  # negatives
                    ),
                    1,
                ).view(-1)
                prepare_time += time.time()

                # and score them
                forward_time -= time.time()
                scores = self.model(
                    triples_to_score[:, 0],
                    triples_to_score[:, 1],
                    triples_to_score[:, 2],
                    comibine='spo',
                    direction="s" if slot == S else ("o" if slot == O else "p"),
                ).view(batch_size, -1)
                #scores = self.model.score_spo(
                #    triples_to_score[:, 0],
                #    triples_to_score[:, 1],
                #    triples_to_score[:, 2],
                #    direction="s" if slot == S else ("o" if slot == O else "p"),
                #).view(batch_size, -1)
                forward_time += time.time()
            elif self._implementation == "sp_po":
                # compute all scores for slot
                forward_time -= time.time()
                if slot == S:
                    all_scores = self.model(None, triples[:, P], triples[:, O], combine='*po')
                    #all_scores = self.model.score_po(triples[:, P], triples[:, O])
                elif slot == O:
                    all_scores = self.model(triples[:, S], triples[:, P], None, combine='sp*')
                    #all_scores = self.model.score_sp(triples[:, S], triples[:, P])
                else:
                    raise NotImplementedError
                forward_time += time.time()

                # determine indexes of relevant scores in scoring matrix
                prepare_time -= time.time()
                row_indexes = (
                    torch.arange(batch_size, device=self.device)
                    .unsqueeze(1)
                    .repeat(1, 1 + num_samples)
                    .view(-1)
                )  # 000 111 222; each 1+num_negative times (here: 3)
                column_indexes = torch.cat(
                    (
                        triples[:, [slot]],  # positives
                        negative_samples[slot],  # negatives
                    ),
                    1,
                ).view(-1)
                prepare_time += time.time()

                # now pick the scores we need
                forward_time -= time.time()
                scores = all_scores[row_indexes, column_indexes].view(batch_size, -1)
                forward_time += time.time()
            else:
                raise ValueError(
                    "invalid implementation: "
                    "negative_sampling.implementation={}".format(self._implementation)
                )

            # compute loss
            forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels, num_negatives=num_samples) / batch_size
            )
            loss_value += loss_value_torch.item()
            forward_time += time.time()

            # backward pass
            backward_time -= time.time()
            loss_value_torch.backward()
            backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp* and *po, treating all other entities as negative."""

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.is_prepared = False
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.num_examples = self.dataset.train().size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {"triples": self.dataset.train()[batch, :].long()},
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.is_prepared = True

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        triples = batch["triples"].to(self.device)
        batch_size = len(triples)
        prepare_time += time.time()

        # forward/backward pass (sp)
        forward_time = -time.time()
        scores_sp = self.model(triples[:, 0], triples[:, 1], None, combine='sp*')
        #scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.model(None, None, None, scores=scores_sp,
                                   labels=triples[:, 2], combine="loss").sum()
        loss_value_sp = loss_value_sp.sum()
        #loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        #loss_value = loss_value_sp.item()
        forward_time = +time.time()
        backward_time = -time.time()
        #self.model(None, None, None, combine="back", loss=loss_value_sp)
        loss_value_sp.backward()
        loss_value = (loss_value_sp / batch_size).item()
        backward_time += time.time()

        # forward/backward pass (po)
        forward_time -= time.time()
        scores_po = self.model(None, triples[:, 1], triples[:, 2], combine='*po')
        #scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.model(None, None, None, scores=scores_po,
                                   labels=triples[:, 0], combine="loss"
                                   ).sum() / batch_size
        #loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        #loss_value += loss_value_po.item()
        loss_value += loss_value_po.item()
        forward_time += time.time()
        backward_time -= time.time()
        loss_value_po.backward()
        backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )
