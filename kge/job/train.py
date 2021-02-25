import itertools
import os
import math
import time
import traceback
from collections import defaultdict

from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from kge.util.io import load_checkpoint
from kge.job.trace import format_trace_entry
from typing import Any, Callable, Dict, List, Optional
import kge.job.util
from kge.util.metric import Metric
from kge.misc import init_from

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


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


class TrainingJob(TrainingOrEvaluationJob):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        if model is None:
            self.model: KgeModel = KgeModel.create(config, dataset)
        else:
            self.model: KgeModel = model
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.batch_size: int = config.get("train.batch_size")
        self._subbatch_auto_tune: bool = config.get("train.subbatch_auto_tune")
        self._max_subbatch_size: int = config.get("train.subbatch_size")
        self.device: str = self.config.get("job.device")
        self.train_split = config.get("train.split")

        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.is_forward_only = forward_only

        if not self.is_forward_only:
            self.model.train()
            self.optimizer = KgeOptimizer.create(config, self.model)
            self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)
            self._lr_warmup = self.config.get("train.lr_warmup")
            for group in self.optimizer.param_groups:
                group["initial_lr"]=group["lr"]

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

        # Hooks run after validation. The corresponding valid trace entry can be found
        # in self.valid_trace[-1] Signature: job
        self.post_valid_hooks: List[Callable[[Job], Any]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        train_type = config.get("train.type")
        class_name = config.get_default(f"{train_type}.class_name")
        return init_from(
            class_name,
            config.modules(),
            config,
            dataset,
            parent_job,
            model=model,
            forward_only=forward_only,
        )

    def _run(self) -> None:
        """Start/resume the training job and run to completion."""

        if self.is_forward_only:
            raise Exception(
                f"{self.__class__.__name__} was initialized for forward only. You can only call run_epoch()"
            )
        if self.epoch == 0:
            self.save(self.config.checkpoint_file(0))

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
                best_index = Metric(self).best_index(
                    list(map(lambda trace: trace[metric_name], self.valid_trace))
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
                    "valid.early_stopping.threshold.epochs"
                ):
                    achieved = self.valid_trace[best_index][metric_name]
                    target = self.config.get(
                        "valid.early_stopping.threshold.metric_value"
                    )
                    if Metric(self).better(target, achieved):
                        self.config.log(
                            "Stopping early ({} did not achieve threshold after {} epochs".format(
                                metric_name, self.epoch
                            )
                        )
                        break

            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break

            # update learning rate if warmup is used
            if self.epoch < self._lr_warmup:
                for group in self.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (self.epoch+1) / self._lr_warmup

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            # validate
            lr_metric = None
            if (
                self.config.get("valid.every") > 0
                and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self)
                self.model.meta["valid_trace_entry"] = trace_entry
                lr_metric = trace_entry[metric_name]

            # update learning rate after warmup
            if self.epoch >= self._lr_warmup:
                # note: lr_metric is None if no validation has been performed in this
                # epoch. This is handled by the optimizers
                self.kge_lr_scheduler.step(lr_metric)

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
                if delete_checkpoint_epoch >= 0:
                    if delete_checkpoint_epoch != 0 or not self.config.get(
                        "train.checkpoint.keep_init"
                    ):
                        self._delete_checkpoint(delete_checkpoint_epoch)

        self.trace(event="train_completed")

    def _delete_checkpoint(self, checkpoint_id):
        """Try to delete checkpoint specified by id"""
        if os.path.exists(self.config.checkpoint_file(checkpoint_id)):
            self.config.log(
                "Removing old checkpoint {}...".format(
                    self.config.checkpoint_file(checkpoint_id)
                )
            )
            os.remove(self.config.checkpoint_file(checkpoint_id))
        else:
            self.config.log(
                "Could not delete old checkpoint {}, does not exist.".format(
                    self.config.checkpoint_file(checkpoint_id)
                )
            )

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        checkpoint = self.save_to({})
        torch.save(
            checkpoint,
            filename,
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
            event="job_resumed",
            epoch=self.epoch,
            checkpoint_file=checkpoint["file"],
        )
        self.config.log(
            "Resuming training from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

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
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = {
                "type": self.type_str,
                "scope": "batch",
                "epoch": self.epoch,
                "split": self.train_split,
                "batch": batch_index,
                "batches": len(self.loader),
            }
            if not self.is_forward_only:
                self.current_trace["batch"].update(
                    lr=[group["lr"] for group in self.optimizer.param_groups],
                )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # process batch (preprocessing + forward pass + backward pass on loss)
            done = False
            while not done:
                try:
                    # try running the batch
                    if not self.is_forward_only:
                        self.optimizer.zero_grad()
                    batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                        batch_index, batch
                    )
                    done = True
                except RuntimeError as e:
                    # is it a CUDA OOM exception and are we allowed to reduce the
                    # subbatch size on such an error? if not, raise the exception again
                    if (
                        "CUDA out of memory" not in str(e)
                        or not self._subbatch_auto_tune
                    ):
                        raise e

                    # try rerunning with smaller subbatch size
                    tb = traceback.format_exc()
                    self.config.log(tb)
                    self.config.log(
                        "Caught OOM exception when running a batch; "
                        "trying to reduce the subbatch size..."
                    )

                    if self._max_subbatch_size <= 0:
                        self._max_subbatch_size = self.batch_size
                    if self._max_subbatch_size <= 1:
                        self.config.log(
                            "Cannot reduce subbatch size "
                            f"(current value: {self._max_subbatch_size})"
                        )
                        raise e  # cannot reduce further

                    self._max_subbatch_size //= 2
                    self.config.set(
                        "train.subbatch_size", self._max_subbatch_size, log=True
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
                if not self.is_forward_only:
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
                        "reserved={:14,} max_allocated={:14,}".format(
                            torch.cuda.memory_allocated(self.device),
                            torch.cuda.memory_reserved(self.device),
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    )

            # update parameters
            batch_optimizer_time = -time.time()
            if not self.is_forward_only:
                self.optimizer.step()
            batch_optimizer_time += time.time()

            # update batch trace with the results
            self.current_trace["batch"].update(
                {
                    "size": batch_result.size,
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                    "event": "batch_completed",
                }
            )

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # print console feedback
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

            # update epoch times
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

        # add results to trace entry
        self.current_trace["epoch"].update(
            dict(
                avg_loss=sum_loss / self.num_examples,
                avg_penalty=sum_penalty / len(self.loader),
                avg_penalties={
                    k: p / len(self.loader) for k, p in sum_penalties.items()
                },
                avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
                epoch_time=epoch_time,
                prepare_time=prepare_time,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=optimizer_time,
                other_time=other_time,
                event="epoch_completed",
            )
        )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None

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

        avg_loss: float = 0.0
        size: int = 0
        prepare_time: float = 0.0
        forward_time: float = 0.0
        backward_time: float = 0.0

    def _process_batch(self, batch_index, batch) -> _ProcessBatchResult:
        "Breaks a batch into subbatches and processes them in turn."
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._process_subbatch(batch_index, batch, subbatch_slice, result)

        return result

    def _prepare_batch(self, batch_index, batch, result: _ProcessBatchResult):
        """Prepare the given batch for processing and determine the batch size.

        batch size must be written into result.size.
        """
        raise NotImplementedError

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: _ProcessBatchResult,
    ):
        """Run forward and backward pass on the given subbatch.

        Also update result.

        """
        raise NotImplementedError
