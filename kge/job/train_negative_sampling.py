import time
import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.util import KgeSampler
from kge.model.transe import TransEScorer

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self.type_str = "negative_sampling"

        if self.__class__ == TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # select negative sampling implementation
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            else:
                self._implementation = "batch"
            self.config.set(
                "negative_sampling.implementation", self._implementation, log=True
            )

        self.config.log(
            "Preparing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )

        # construct dataloader
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

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()
