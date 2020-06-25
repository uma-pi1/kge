import itertools
import os
import math
import time
import sys
from collections import defaultdict

from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from kge.util.io import load_checkpoint
from typing import Any, Callable, Dict, List, Optional, Union
import kge.job.util

SLOTS = [0, 1, 2]
S, P, O = SLOTS


def _generate_worker_init_fn(config):
    "Initialize workers of a DataLoader"
    use_fixed_seed = config.get("random_seed.numpy") >= 0

    def worker_init_fn(worker_num):
        # ensure that NumPy uses different seeds at each worker
        if use_fixed_seed:
            # reseed based on current seed (same for all workers) and worker number
            # (different)
            base_seed = np.random.randint(2 ** 32 - 1)
            np.random.seed(base_seed + worker_num)
        else:
            # reseed fresh
            np.random.seed()

    return worker_init_fn


class TrainingJob(Job):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self, config: Config, dataset: Dataset, parent_job: Job = None, model=None
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        if model is None:
            self.model: KgeModel = KgeModel.create(config, dataset)
        else:
            self.model: KgeModel = model
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.batch_size: int = config.get("train.batch_size")
        self.device: str = self.config.get("job.device")
        self.train_split = config.get("train.split")

        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.valid_trace: List[Dict[str, Any]] = []
        valid_conf = config.clone()
        valid_conf.set("job.type", "eval")
        if self.config.get("valid.split") != "":
            valid_conf.set("eval.split", self.config.get("valid.split"))
        valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
        self.valid_job = EvaluationJob.create(
            valid_conf, dataset, parent_job=self, model=self.model
        )

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks: List[Callable[[Job], Any]] = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run after a validation job.
        #: Signature: job, trace_entry
        self.post_valid_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run after training
        #: Signature: job, trace_entry
        self.post_train_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

        self.model.train()

    @staticmethod
    def create(
        config: Config, dataset: Dataset, parent_job: Job = None, model=None
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "KvsAll":
            return TrainingJobKvsAll(config, dataset, parent_job, model=model)
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(config, dataset, parent_job, model=model)
        elif config.get("train.type") == "1vsAll":
            return TrainingJob1vsAll(config, dataset, parent_job, model=model)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def _run(self) -> None:
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
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            # validate and update learning rate
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
                self.kge_lr_scheduler.step(trace_entry[metric_name])
            else:
                self.kge_lr_scheduler.step()

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
        checkpoint = self.save_to({})
        torch.save(
            checkpoint, filename,
        )

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds trainjob specific information to the checkpoint"""
        train_checkpoint = {
            "type": "train",
            "epoch": self.epoch,
            "valid_trace": self.valid_trace,
            "model": self.model.save(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
            "job_id": self.job_id,
        }
        train_checkpoint = self.config.save_to(train_checkpoint)
        checkpoint.update(train_checkpoint)
        return checkpoint

    def _load(self, checkpoint: Dict) -> str:
        if checkpoint["type"] != "train":
            raise ValueError("Training can only be continued on trained checkpoints")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            # new format
            self.kge_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint["file"],
        )
        self.config.log(
            "Resuming training from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    def run_epoch(self) -> Dict[str, Any]:
        "Runs an epoch and returns a trace entry."

        # variables that record various statitics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = defaultdict(lambda: 0.0)
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
            penalties_torch = self.model.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                sum_penalties[penalty_key] += penalty_value_torch.item()
            sum_penalty += penalty
            batch_backward_time += time.time()

            # determine full cost
            cost_value = batch_result.avg_loss + penalty

            # abort on nan
            if self.abort_on_nan and math.isnan(cost_value):
                raise FloatingPointError("Cost became nan, aborting training job")

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
            self.optimizer.step()
            batch_optimizer_time += time.time()

            # tracing/logging
            if self.trace_batch:
                batch_trace = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "split": self.train_split,
                    "batch": batch_index,
                    "size": batch_result.size,
                    "batches": len(self.loader),
                    "lr": [group["lr"] for group in self.optimizer.param_groups],
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
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
            self.config.print(
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
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )
        trace_entry = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
            lr=[group["lr"] for group in self.optimizer.param_groups],
            avg_loss=sum_loss / self.num_examples,
            avg_penalty=sum_penalty / len(self.loader),
            avg_penalties={k: p / len(self.loader) for k, p in sum_penalties.items()},
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
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

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
    """Train with examples consisting of a query and its answers.

    Terminology:
    - Query type: which queries to ask (sp_, s_o, and/or _po), can be configured via
      configuration key `KvsAll.query_type` (which see)
    - Query: a particular query, e.g., (John,marriedTo) of type sp_
    - Labels: list of true answers of a query (e.g., [Jane])
    - Example: a query + its labels, e.g., (John,marriedTo), [Jane]
    """

    from kge.indexing import KvsAllIndex

    def __init__(self, config, dataset, parent_job=None, model=None):
        super().__init__(config, dataset, parent_job, model=model)
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
        super()._prepare()
        # determine enabled query types
        self.query_types = [
            key
            for key, enabled in self.config.get("KvsAll.query_types").items()
            if enabled
        ]

        # corresponding indexes
        self.query_indexes: List[KvsAllIndex] = []

        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        self.query_last_example = []

        # construct relevant data structures
        self.num_examples = 0
        for query_type in self.query_types:
            index_type = (
                "sp_to_o"
                if query_type == "sp_"
                else ("so_to_p" if query_type == "s_o" else "po_to_s")
            )
            index = self.dataset.index(f"{self.train_split}_{index_type}")
            self.query_indexes.append(index)
            self.num_examples += len(index)
            self.query_last_example.append(self.num_examples)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a dictionary of:

            - queries: nx2 tensor, row = query (sp, po, or so indexes)
            - label_coords: for each query, position of true answers (an Nx2 tensor,
              first columns holds query index, second colum holds index of label)
            - query_type_indexes (vector of size n holding the query type of each query)
            - triples (all true triples in the batch; e.g., needed for weighted
              penalties)

            """

            # count how many labels we have across the entire batch
            num_ones = 0
            for example_index in batch:
                start = 0
                for query_type_index in range(len(self.query_types)):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        num_ones += self.query_indexes[query_type_index]._values_offset[
                            example_index + 1
                        ]
                        num_ones -= self.query_indexes[query_type_index]._values_offset[
                            example_index
                        ]
                        break
                    start = end

            # now create the batch elements
            queries_batch = torch.zeros([len(batch), 2], dtype=torch.long)
            query_type_indexes_batch = torch.zeros([len(batch)], dtype=torch.long)
            label_coords_batch = torch.zeros([num_ones, 2], dtype=torch.int)
            triples_batch = torch.zeros([num_ones, 3], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                start = 0
                for query_type_index, query_type in enumerate(self.query_types):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        query_type_indexes_batch[batch_index] = query_type_index
                        queries = self.query_indexes[query_type_index]._keys
                        label_offsets = self.query_indexes[
                            query_type_index
                        ]._values_offset
                        labels = self.query_indexes[query_type_index]._values
                        if query_type == "sp_":
                            query_col_1, query_col_2, target_col = S, P, O
                        elif query_type == "s_o":
                            query_col_1, target_col, query_col_2 = S, P, O
                        else:
                            target_col, query_col_1, query_col_2 = S, P, O
                        break
                    start = end

                queries_batch[batch_index,] = queries[example_index]
                start = label_offsets[example_index]
                end = label_offsets[example_index + 1]
                size = end - start
                label_coords_batch[
                    current_index : (current_index + size), 0
                ] = batch_index
                label_coords_batch[current_index : (current_index + size), 1] = labels[
                    start:end
                ]
                triples_batch[
                    current_index : (current_index + size), query_col_1
                ] = queries[example_index][0]
                triples_batch[
                    current_index : (current_index + size), query_col_2
                ] = queries[example_index][1]
                triples_batch[
                    current_index : (current_index + size), target_col
                ] = labels[start:end]
                current_index += size

            # all done
            return {
                "queries": queries_batch,
                "label_coords": label_coords_batch,
                "query_type_indexes": query_type_indexes_batch,
                "triples": triples_batch,
            }

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        queries_batch = batch["queries"].to(self.device)
        batch_size = len(queries_batch)
        label_coords_batch = batch["label_coords"].to(self.device)
        query_type_indexes_batch = batch["query_type_indexes"]

        # in this method, example refers to the index of an example in the batch, i.e.,
        # it takes values in 0,1,...,batch_size-1
        examples_for_query_type = {}
        for query_type_index, query_type in enumerate(self.query_types):
            examples_for_query_type[query_type] = (
                (query_type_indexes_batch == query_type_index)
                .nonzero()
                .to(self.device)
                .view(-1)
            )

        labels_batch = kge.job.util.coord_to_sparse_tensor(
            batch_size,
            max(self.dataset.num_entities(), self.dataset.num_relations()),
            label_coords_batch,
            self.device,
        ).to_dense()
        labels_for_query_type = {}
        for query_type, examples in examples_for_query_type.items():
            if query_type == "s_o":
                labels_for_query_type[query_type] = labels_batch[
                    examples, : self.dataset.num_relations()
                ]
            else:
                labels_for_query_type[query_type] = labels_batch[
                    examples, : self.dataset.num_entities()
                ]

        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            for query_type, labels in labels_for_query_type.items():
                if query_type != "s_o":  # entity targets only for now
                    labels_for_query_type[query_type] = (
                        1.0 - self.label_smoothing
                    ) * labels + 1.0 / labels.size(1)

        prepare_time += time.time()

        # forward/backward pass (sp)
        loss_value_total = 0.0
        backward_time = 0
        forward_time = 0
        for query_type, examples in examples_for_query_type.items():
            if len(examples) > 0:
                forward_time -= time.time()
                if query_type == "sp_":
                    scores = self.model.score_sp(
                        queries_batch[examples, 0], queries_batch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.model.score_so(
                        queries_batch[examples, 0], queries_batch[examples, 1]
                    )
                else:
                    scores = self.model.score_po(
                        queries_batch[examples, 0], queries_batch[examples, 1]
                    )
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type]) / batch_size
                )
                loss_value_total = loss_value.item()
                forward_time += time.time()
                backward_time -= time.time()
                loss_value.backward()
                backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value_total, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(self, config, dataset, parent_job=None, model=None):
        super().__init__(config, dataset, parent_job, model=model)
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            elif max_nr_of_negs > 30:
                self._implementation = "batch"
        self._max_chunk_size = self.config.get("negative_sampling.chunk_size")

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
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.split(self.train_split)[batch, :].long()
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
        batch_triples = batch["triples"].to(self.device)
        batch_negative_samples = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]
        batch_size = len(batch_triples)
        prepare_time += time.time()

        loss_value = 0.0
        forward_time = 0.0
        backward_time = 0.0
        labels = None

        # perform processing of batch in smaller chunks to save memory
        max_chunk_size = (
            self._max_chunk_size if self._max_chunk_size > 0 else batch_size
        )
        for chunk_number in range(math.ceil(batch_size / max_chunk_size)):
            # determine data used for this chunk
            chunk_start = max_chunk_size * chunk_number
            chunk_end = min(max_chunk_size * (chunk_number + 1), batch_size)
            negative_samples = [
                ns[chunk_start:chunk_end, :] for ns in batch_negative_samples
            ]
            triples = batch_triples[chunk_start:chunk_end, :]
            chunk_size = chunk_end - chunk_start

            # process the chunk
            for slot in [S, P, O]:
                num_samples = self._sampler.num_samples[slot]
                if num_samples <= 0:
                    continue

                # construct gold labels: first column corresponds to positives,
                # remaining columns to negatives
                if labels is None or labels.shape != torch.Size(
                    [chunk_size, 1 + num_samples]
                ):
                    prepare_time -= time.time()
                    labels = torch.zeros(
                        (chunk_size, 1 + num_samples), device=self.device
                    )
                    labels[:, 0] = 1
                    prepare_time += time.time()

                # compute corresponding scores
                scores = None
                if self._implementation == "triple":
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
                    scores = self.model.score_spo(
                        triples_to_score[:, 0],
                        triples_to_score[:, 1],
                        triples_to_score[:, 2],
                        direction="s" if slot == S else ("o" if slot == O else "p"),
                    ).view(chunk_size, -1)
                    forward_time += time.time()
                elif self._implementation == "all":
                    # Score against all possible targets. Creates a score matrix of size
                    # [chunk_size, num_entities] or [chunk_size, num_relations]. All
                    # scores relevant for positive and negative triples are contained in
                    # this score matrix.

                    # compute all scores for slot
                    forward_time -= time.time()
                    if slot == S:
                        all_scores = self.model.score_po(triples[:, P], triples[:, O])
                    elif slot == P:
                        all_scores = self.model.score_so(triples[:, S], triples[:, O])
                    elif slot == O:
                        all_scores = self.model.score_sp(triples[:, S], triples[:, P])
                    else:
                        raise NotImplementedError
                    forward_time += time.time()

                    # determine indexes of relevant scores in scoring matrix
                    prepare_time -= time.time()
                    row_indexes = (
                        torch.arange(chunk_size, device=self.device)
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
                    scores = all_scores[row_indexes, column_indexes].view(
                        chunk_size, -1
                    )
                    forward_time += time.time()
                elif self._implementation == "batch":
                    # Score against all targets contained in the chunk. Creates a score
                    # matrix of size [chunk_size, unique_entities_in_slot] or
                    # [chunk_size, unique_relations_in_slot]. All scores
                    # relevant for positive and negative triples are contained in this
                    # score matrix.
                    forward_time -= time.time()
                    unique_targets, column_indexes = torch.unique(
                        torch.cat((triples[:, [slot]], negative_samples[slot]), 1).view(
                            -1
                        ),
                        return_inverse=True,
                    )

                    # compute scores for all unique targets for slot
                    if slot == S:
                        all_scores = self.model.score_po(
                            triples[:, P], triples[:, O], unique_targets
                        )
                    elif slot == P:
                        all_scores = self.model.score_so(
                            triples[:, S], triples[:, O], unique_targets
                        )
                    elif slot == O:
                        all_scores = self.model.score_sp(
                            triples[:, S], triples[:, P], unique_targets
                        )
                    else:
                        raise NotImplementedError
                    forward_time += time.time()

                    # determine indexes of relevant scores in scoring matrix
                    prepare_time -= time.time()
                    row_indexes = (
                        torch.arange(chunk_size, device=self.device)
                        .unsqueeze(1)
                        .repeat(1, 1 + num_samples)
                        .view(-1)
                    )  # 000 111 222; each 1+num_negative times (here: 3)
                    prepare_time += time.time()

                    # now pick the scores we need
                    forward_time -= time.time()
                    scores = all_scores[row_indexes, column_indexes].view(
                        chunk_size, -1
                    )
                    forward_time += time.time()

                # compute chunk loss (concluding the forward pass of the chunk)
                forward_time -= time.time()
                loss_value_torch = (
                    self.loss(scores, labels, num_negatives=num_samples) / batch_size
                )
                loss_value += loss_value_torch.item()
                forward_time += time.time()

                # backward pass for this chunk
                backward_time -= time.time()
                loss_value_torch.backward()
                backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(self, config, dataset, parent_job=None, model=None):
        super().__init__(config, dataset, parent_job, model=model)
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )


    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        triples = batch["triples"].to(self.device)
        batch_size = len(triples)
        prepare_time += time.time()

        # forward/backward pass (sp)
        forward_time = -time.time()
        scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        loss_value = loss_value_sp.item()
        forward_time += time.time()
        backward_time = -time.time()
        loss_value_sp.backward()
        backward_time += time.time()

        # forward/backward pass (po)
        forward_time -= time.time()
        scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        loss_value += loss_value_po.item()
        forward_time += time.time()
        backward_time -= time.time()
        loss_value_po.backward()
        backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )
