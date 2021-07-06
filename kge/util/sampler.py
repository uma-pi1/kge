from kge import Config, Configurable, Dataset
from kge.indexing import where_in

import random
import torch
from typing import Optional
import numpy as np
import numba
import time

SLOTS = [0, 1, 2]
SLOT_STR = ["s", "p", "o"]
S, P, O = SLOTS


class KgeSampler(Configurable):
    """Negative sampler. """

    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key)

        # load config
        self.num_samples = torch.zeros(3, dtype=torch.int)
        self.filter_positives = torch.zeros(3, dtype=torch.bool)
        self.vocabulary_size = torch.zeros(3, dtype=torch.int)
        self.shared = self.get_option("shared")
        self.shared_type = self.check_option("shared_type", ["naive", "default"])
        self.with_replacement = self.get_option("with_replacement")
        if not self.with_replacement and not self.shared:
            raise ValueError(
                "Without replacement sampling is only supported when "
                "shared negative sampling is enabled."
            )
        self.filtering_split = config.get("negative_sampling.filtering.split")
        if self.filtering_split == "":
            self.filtering_split = config.get("train.split")
        for slot in SLOTS:
            slot_str = SLOT_STR[slot]
            self.num_samples[slot] = self.get_option(f"num_samples.{slot_str}")
            self.filter_positives[slot] = self.get_option(f"filtering.{slot_str}")
            self.vocabulary_size[slot] = (
                dataset.num_relations() if slot == P else dataset.num_entities()
            )
            # create indices for filtering here already if needed and not existing
            # otherwise every worker would create every index again and again
            if self.filter_positives[slot]:
                pair = ["po", "so", "sp"][slot]
                dataset.index(f"{self.filtering_split}_{pair}_to_{slot_str}")
        if any(self.filter_positives):
            if self.shared:
                raise ValueError(
                    "Filtering is not supported when shared negative sampling is enabled."
                )
            self.filter_implementation = self.check_option(
                "filtering.implementation", ["standard", "fast", "fast_if_available"]
            )
        self.dataset = dataset
        # auto config
        for slot, copy_from in [(S, O), (P, None), (O, S)]:
            if self.num_samples[slot] < 0:
                if copy_from is not None and self.num_samples[copy_from] > 0:
                    self.num_samples[slot] = self.num_samples[copy_from]
                else:
                    self.num_samples[slot] = 0

    @staticmethod
    def create(
        config: Config, configuration_key: str, dataset: Dataset
    ) -> "KgeSampler":
        """Factory method for sampler creation."""
        sampling_type = config.get(configuration_key + ".sampling_type")
        if sampling_type == "uniform":
            return KgeUniformSampler(config, configuration_key, dataset)
        elif sampling_type == "frequency":
            return KgeFrequencySampler(config, configuration_key, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(configuration_key + ".sampling_type")

    def sample(
        self,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: Optional[int] = None,
    ) -> "BatchNegativeSample":
        """Obtain a set of negative samples for a specified slot.

        `positive_triples` is a batch_size x 3 tensor of positive triples. `slot` is
        either 0 (subject), 1 (predicate), or 2 (object). If `num_samples` is `None`,
        it is set to the default value for the slot configured in this sampler.

        Returns a `BatchNegativeSample` data structure that allows to retrieve or score
        all negative samples. In the simplest setting, this data structure holds a
        batch_size x num_samples tensor with the negative sample indexes (see
        `DefaultBatchNegativeSample`), but more efficient approaches may be used by
        certain samplers.

        """
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if self.shared:
            # for shared sampling, we do not post-process; return right away
            return self._sample_shared(positive_triples, slot, num_samples)
        else:
            negative_samples = self._sample(positive_triples, slot, num_samples)

        # for non-shared smaples, we filter the positives (if set in config)
        if self.filter_positives[slot]:
            if self.filter_implementation == "fast":
                negative_samples = self._filter_and_resample_fast(
                    negative_samples, slot, positive_triples
                )
            elif self.filter_implementation == "standard":
                negative_samples = self._filter_and_resample(
                    negative_samples, slot, positive_triples
                )
            else:  # fast_if_available
                try:
                    negative_samples = self._filter_and_resample_fast(
                        negative_samples, slot, positive_triples
                    )
                    self.filter_implementation = "fast"
                except NotImplementedError:
                    negative_samples = self._filter_and_resample(
                        negative_samples, slot, positive_triples
                    )
                    self.filter_implementation = "standard"

        return DefaultBatchNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            negative_samples,
        )

    def _sample(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> torch.Tensor:
        """Sample negative examples.

        This methods returns a tensor of size batch_size x num_samples holding the
        indexes for the sample. The method is also used to resample filtered positives.

        """
        raise NotImplementedError("The selected sampler is not implemented.")

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> "BatchNegativeSample":
        """Sample negative examples with sharing.

        This methods directly returns a BatchNegativeSample data structure for
        efficiency.

        """
        raise NotImplementedError(
            "The selected sampler does not support shared negative samples."
        )

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        """Filter and resample indices until only negatives have been created. """
        pair_str = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(
            f"{self.filtering_split}_{pair_str}_to_{SLOT_STR[slot]}"
        )
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols]
        for i in range(positive_triples.size(0)):
            positives = index.get((pairs[i][0].item(), pairs[i][1].item())).numpy()
            # indices of samples that have to be sampled again
            resample_idx = where_in(negative_samples[i].numpy(), positives)
            # number of new samples needed
            num_new = len(resample_idx)
            # number already found of the new samples needed
            num_found = 0
            num_remaining = num_new - num_found
            while num_remaining:
                new_samples = self._sample(
                    positive_triples[i, None], slot, num_remaining
                ).view(-1)
                # indices of the true negatives
                tn_idx = where_in(new_samples.numpy(), positives, not_in=True)
                # write the true negatives found
                if len(tn_idx):
                    negative_samples[
                        i, resample_idx[num_found : num_found + len(tn_idx)]
                    ] = new_samples[tn_idx]
                    num_found += len(tn_idx)
                    num_remaining = num_new - num_found
        return negative_samples

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        """Filter and resample indices.

        Samplers can override this method when their sampling strategy allows for a
        more efficient filtering method than the generic standard method or when their
        code can be optimized by tools such as Numba.

        """
        raise NotImplementedError(
            "Use filtering.implementation=standard for this sampler."
        )


class BatchNegativeSample(Configurable):
    """Abstract superclass for a negative sample of a batch.

    Provides methods to access the negative samples and to score them using a model.
    """

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
    ):
        super().__init__(config, configuration_key)
        self.positive_triples = positive_triples
        self.slot = slot
        self.num_samples = num_samples
        self._implementation = self.check_option(
            "implementation", ["triple", "batch", "all"]
        )
        self.forward_time = 0.0
        self.prepare_time = 0.0

    def samples(self, indexes=None) -> torch.Tensor:
        """Returns a tensor holding the indexes of the negative samples.

        If `indexes` is provided, only score the corresponding subset of the batch.

        Returns a chunk_size x num_samples tensor of indexes. Here chunk_size corresponds
        the batch size (if `indexes=None`) or to the number of specified indexes (otherwise).
        """
        raise NotImplementedError

    def unique_samples(self, indexes=None, return_inverse=False):
        """Returns the unique negative samples.

        If `indexes` is provided, only consider the corresponding subset of the batch.
        Optionally, also returns the indexes of each unqiue sample in the flattened
        negative-sampling tensor (i.e., in `self.samples(indexes).view(-1)`).

        """
        samples = self.samples(indexes)
        return torch.unique(samples.view(-1), return_inverse=return_inverse)

    def to(self, device) -> "BatchNegativeSample":
        """Move the negative samples to the specified device."""
        self.positive_triples = self.positive_triples.to(device)
        return self

    def score(self, model, indexes=None) -> torch.Tensor:
        """Score the negative samples for the batch with the provided model.

        If `indexes` is provided, only score the corresponding subset of the batch.

        Returns a chunk_size x num_samples tensor of scores. Here chunk_size corresponds
        the batch size (if `indexes=None`) or to the number of specified indexes (otherwise).

        Sets the `forward_time` and `prepare_time` attributes.
        """
        self.forward_time = 0.0
        self.prepare_time = 0.0

        # the default implementation here is based on the set of all samples as provided
        # by self.samples(); get the relavant data
        slot = self.slot
        self.prepare_time -= time.time()
        negative_samples = self.samples(indexes)
        num_samples = self.num_samples
        triples = (
            self.positive_triples[indexes, :] if indexes else self.positive_triples
        )
        self.prepare_time += time.time()

        # go ahead and score
        device = self.positive_triples.device
        chunk_size = len(negative_samples)
        scores = None
        if self._implementation == "triple":
            # construct triples
            self.prepare_time -= time.time()
            triples_to_score = triples.repeat(1, num_samples).view(-1, 3)
            triples_to_score[:, slot] = negative_samples.contiguous().view(-1)
            self.prepare_time += time.time()

            # and score them
            self.forward_time -= time.time()
            scores = model.score_spo(
                triples_to_score[:, S],
                triples_to_score[:, P],
                triples_to_score[:, O],
                direction=SLOT_STR[slot],
            ).view(chunk_size, -1)
            self.forward_time += time.time()
        elif self._implementation in ["batch", "all"]:
            # Score each triples against all unique possible targets, then pick out the
            # actual scores.
            self.prepare_time -= time.time()
            if self._implementation == "all":
                unique_targets = None  # means all
                column_indexes = negative_samples.contiguous().view(-1)
            else:
                unique_targets, column_indexes = self.unique_samples(
                    indexes, return_inverse=True
                )
            self.prepare_time += time.time()

            # compute all scores for slot
            self.forward_time -= time.time()
            all_scores = self._score_unique_targets(
                model, slot, triples, unique_targets
            )
            self.forward_time += time.time()

            # determine indexes of relevant scores in scoring matrix
            self.prepare_time -= time.time()
            row_indexes = (
                torch.arange(chunk_size, device=device)
                .unsqueeze(1)
                .repeat(1, num_samples)
                .view(-1)
            )  # 000 111 222; each num_samples times (here: 3)
            self.prepare_time += time.time()

            # and pick the scores we need
            self.forward_time -= time.time()
            scores = all_scores[row_indexes, column_indexes].view(chunk_size, -1)
            self.forward_time += time.time()
        else:
            raise ValueError

        return scores

    @staticmethod
    def _score_unique_targets(model, slot, triples, unique_targets) -> torch.Tensor:
        if slot == S:
            all_scores = model.score_po(triples[:, P], triples[:, O], unique_targets)
        elif slot == P:
            all_scores = model.score_so(triples[:, S], triples[:, O], unique_targets)
        elif slot == O:
            all_scores = model.score_sp(triples[:, S], triples[:, P], unique_targets)
        else:
            raise NotImplementedError
        return all_scores


class DefaultBatchNegativeSample(BatchNegativeSample):
    """Default implementation that stores all negative samples as a tensor."""

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        samples: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._samples = samples

    def samples(self, indexes=None) -> torch.Tensor:
        return self._samples if indexes is None else self._samples[indexes]

    def to(self, device) -> "DefaultBatchNegativeSample":
        super().to(device)
        self._samples = self._samples.to(device)
        return self


class NaiveSharedNegativeSample(BatchNegativeSample):
    """Implementation for naive shared sampling.

    Here all triples use exactly the same negatives samples.

    """

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        unique_samples: torch.Tensor,
        repeat_indexes: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._unique_samples = unique_samples
        self._repeat_indexes = repeat_indexes

    def unique_samples(self, indexes=None, return_inverse=False) -> torch.Tensor:
        if return_inverse:
            # slow but probably rarely used anyway
            samples = self.samples(indexes)
            return torch.unique(samples.contiguous().view(-1), return_inverse=True)
        else:
            return self._unique_samples

    def samples(self, indexes=None) -> torch.Tensor:
        # create one row, then expand to chunk size
        chunk_size = len(indexes) if indexes else len(self.positive_triples)
        device = self.positive_triples.device
        num_unique = len(self._unique_samples)
        if num_unique == self.num_samples:
            negative_samples1 = self._unique_samples
        else:
            negative_samples1 = torch.empty(
                self.num_samples, dtype=torch.long, device=device
            )
            negative_samples1[:num_unique] = self._unique_samples
            negative_samples1[num_unique:] = self._unique_samples[self._repeat_indexes]

        return negative_samples1.unsqueeze(0).expand((chunk_size, -1))

    def score(self, model, indexes=None) -> torch.Tensor:
        if self._implementation != "batch":
            return super().score(model, indexes)

        # for batch, we have a faster implementation that avoids creating the full
        # sample tensor
        self.prepare_time = 0.0
        self.forward_time = 0.0
        slot = self.slot
        unique_targets = self._unique_samples
        num_unique = len(unique_targets)
        triples = (
            self.positive_triples
            if indexes is None
            else self.positive_triples[indexes, :]
        )
        chunk_size = len(triples)

        # compute scores for all unique targets for slot
        self.forward_time -= time.time()
        scores = self._score_unique_targets(model, slot, triples, unique_targets)

        # repeat scores as needed for WR sampling
        if num_unique != self.num_samples:
            scores = scores[
                :,
                torch.cat(
                    (
                        torch.arange(num_unique, device=scores.device),
                        self._repeat_indexes,
                    )
                ),
            ]
        self.forward_time += time.time()

        return scores

    def to(self, device) -> "NaiveSharedNegativeSample":
        super().to(device)
        self._unique_samples = self._unique_samples.to(device)
        self._repeat_indexes = self._repeat_indexes.to(device)
        return self


class DefaultSharedNegativeSample(BatchNegativeSample):
    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        unique_samples: torch.Tensor,
        drop_index: torch.Tensor,
        repeat_indexes: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._unique_samples = unique_samples
        self._drop_index = drop_index
        self._repeat_indexes = repeat_indexes

    def unique_samples(self, indexes=None, return_inverse=False) -> torch.Tensor:
        if return_inverse:
            # slow but probably rarely used anyway
            return super(DefaultSharedNegativeSample, self).unique_samples(
                indexes=indexes, return_inverse=return_inverse
            )
        drop_index = self._drop_index if indexes is None else self._drop_index[indexes]
        if torch.all(drop_index == drop_index[0]).item():
            # same sample dropped for every triple in the batch
            not_drop_mask = torch.ones(len(self._unique_samples), dtype=torch.bool)
            not_drop_mask[drop_index[0]] = False
            return self._unique_samples[not_drop_mask]
        return self._unique_samples

    def samples(self, indexes=None) -> torch.Tensor:
        num_samples = self.num_samples
        triples = (
            self.positive_triples
            if indexes is None
            else self.positive_triples[indexes, :]
        )
        drop_index = self._drop_index if indexes is None else self._drop_index[indexes]
        chunk_size = len(triples)

        # create output tensor
        device = self.positive_triples.device
        num_unique = len(self._unique_samples) - 1
        negative_samples = torch.empty(
            chunk_size, num_unique, dtype=torch.long, device=device
        )

        # Add the first num_distinct samples for each positive. Dropping is
        # performed by copying the last shared sample over the dropped sample
        negative_samples[:, :] = self._unique_samples[:-1]
        drop_rows = torch.nonzero(drop_index != num_unique, as_tuple=False).squeeze()
        negative_samples[drop_rows, drop_index[drop_rows]] = self._unique_samples[-1]

        # repeat indexes as needed for WR sampling
        if num_unique != num_samples:
            negative_samples = negative_samples[
                :,
                torch.cat(
                    (torch.arange(num_unique, device=device), self._repeat_indexes)
                ),
            ]

        return negative_samples

    def score(self, model, indexes=None) -> torch.Tensor:
        if self._implementation != "batch":
            return super().score(model, indexes)

        # for batch, we have a faster implementation that avoids creating the full
        # sample tensor
        self.prepare_time = 0.0
        self.forward_time = 0.0
        slot = self.slot
        unique_targets = self._unique_samples
        num_unique = len(unique_targets) - 1
        triples = (
            self.positive_triples[indexes, :] if indexes else self.positive_triples
        )
        drop_index = self._drop_index[indexes] if indexes else self._drop_index
        drop_rows = torch.nonzero(drop_index != num_unique, as_tuple=False).squeeze()
        chunk_size = len(triples)

        # compute scores for all unique targets for slot
        self.forward_time -= time.time()
        all_scores = self._score_unique_targets(model, slot, triples, unique_targets)

        # create the complete scoring matrix
        device = self.positive_triples.device
        scores = torch.empty(chunk_size, num_unique, device=device)

        # fill in the unique negative scores. first column is left empty
        # to hold positive scores
        scores[:, :] = all_scores[:, :-1]
        scores[drop_rows, drop_index[drop_rows]] = all_scores[drop_rows, -1]

        # repeat scores as needed for WR sampling
        if num_unique != self.num_samples:
            scores = scores[
                :,
                torch.cat(
                    (torch.arange(num_unique, device=device), self._repeat_indexes)
                ),
            ]
        self.forward_time += time.time()

        return scores

    def to(self, device):
        super().to(device)
        self._unique_samples = self._unique_samples.to(device)
        self._drop_index = self._drop_index.to(device)
        self._repeat_indexes = self._repeat_indexes.to(device)
        return self


class KgeUniformSampler(KgeSampler):
    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key, dataset)

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        return torch.randint(
            self.vocabulary_size[slot], (positive_triples.size(0), num_samples)
        )

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        # For shared_type=naive, produces:
        # - a tensor `unique_samples` of size U holding a list of unique negative
        # samples.
        # - a tensor `repeat_indexes` of size `num_samples-U` holding the indexes of
        #   repeated unique samples
        #
        # For shared_type=default, additionally produces:
        # - one more negative sample in `unique_samples`
        # - a tensor `drop_index` that indicates for each positive triple, which unique
        #   sample is not used for that positive. The dropped sample should be replaced
        #   with the last entry in `unique_samples`. The option to drop a negative
        #   sample is used to avoid using the true positive from `positive_triples` as a
        #   negative sample: when that true positive is `unique_samples`, it should be
        #   ignored.
        #
        # In both case, the data structures are wrapped in the corresponding subclass of
        # BatchNegativeSample.
        batch_size = len(positive_triples)

        # determine number of distinct negative samples for each positive
        if self.with_replacement:
            # Simple way to get a sample from the distribution of number of distinct
            # values in the negative sample (for "default" type: WR sampling except the
            # positive, hence the - 1)
            num_unique = len(
                np.unique(
                    np.random.choice(
                        self.vocabulary_size[slot]
                        if self.shared_type == "naive"
                        else self.vocabulary_size[slot] - 1,
                        num_samples,
                        replace=True,
                    )
                )
            )
        else:  # WOR -> all samples distinct
            num_unique = num_samples

        # Take the WOR sample. For default, take one more WOR sample than necessary
        # (used to replace sampled positives). Numpy is horribly slow for large
        # vocabulary sizes, so we use random.sample instead.
        #
        # SLOW:
        # unique_samples = np.random.choice(
        #     self.vocabulary_size[slot], num_unique, replace=False
        # )
        unique_samples = random.sample(
            range(self.vocabulary_size[slot]),
            num_unique if self.shared_type == "naive" else num_unique + 1,
        )

        # For WR, we need to upsample. To do so, we compute the set of additional
        # (repeated) sample indexes.
        if num_unique != num_samples:  # only happens with WR
            repeat_indexes = torch.tensor(
                np.random.choice(num_unique, num_samples - num_unique, replace=True)
            )
        else:
            repeat_indexes = torch.empty(0)  # WOR or WR when all samples unique

        # for naive shared sampling, we are done
        if self.shared_type == "naive":
            return NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                positive_triples,
                slot,
                num_samples,
                torch.tensor(unique_samples, dtype=torch.long),
                repeat_indexes,
            )

        # For default, we now filter the positives. For each row i (positive triple),
        # select a sample to drop. For rows that contain its positive as a negative
        # example, drop that positive. For all other rows, drop a random position. Here
        # we start with random drop position for each row and then update the ones that
        # contain its positive in the negative samples
        positives = positive_triples[:, slot].numpy()
        drop_index = np.random.choice(num_unique + 1, batch_size, replace=True)
        # TODO can we do the following quicker?
        unique_samples_index = {s: j for j, s in enumerate(unique_samples)}
        for i, v in [
            (i, unique_samples_index.get(positives[i]))
            for i in range(batch_size)
            if positives[i] in unique_samples_index
        ]:
            drop_index[i] = v

        # now we are done for default
        return DefaultSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            torch.tensor(unique_samples, dtype=torch.long),
            torch.tensor(drop_index),
            repeat_indexes,
        )

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        pair_str = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(
            f"{self.filtering_split}_{pair_str}_to_{SLOT_STR[slot]}"
        )
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols].numpy().astype(np.int32)
        batch_size = positive_triples.size(0)
        voc_size = self.vocabulary_size[slot]
        # filling a numba-dict here and then call the function was faster than 1. Using
        # numba lists 2. Using a python list and convert it to an np.array and use
        # offsets 3. Growing a np.array with np.append 4. leaving the loop in python and
        # calling a numba function within the loop
        positives_index = numba.typed.Dict()
        for i in range(batch_size):
            pair = (pairs[i][0], pairs[i][1])
            positives_index[pair] = index.get(pair).numpy()
        negative_samples = negative_samples.numpy()
        KgeUniformSampler._filter_and_resample_numba(
            negative_samples, pairs, positives_index, batch_size, int(voc_size),
        )
        return torch.tensor(negative_samples, dtype=torch.int64)

    @numba.njit
    def _filter_and_resample_numba(
        negative_samples, pairs, positives_index, batch_size, voc_size
    ):
        for i in range(batch_size):
            positives = positives_index[(pairs[i][0], pairs[i][1])]
            # inlining the where_in function here results in an internal numba
            # error which asks to file a bug report
            resample_idx = where_in(negative_samples[i], positives)
            # number of new samples needed
            num_new = len(resample_idx)
            # number already found of the new samples needed
            num_found = 0
            num_remaining = num_new - num_found
            while num_remaining:
                new_samples = np.random.randint(0, voc_size, num_remaining)
                idx = where_in(new_samples, positives, not_in=True)
                # write the true negatives found
                if len(idx):
                    ctr = 0
                    # numba does not support advanced indexing but the loop
                    # is optimized so it's faster than numpy anyway
                    for j in resample_idx[num_found : num_found + len(idx)]:
                        negative_samples[i, j] = new_samples[ctr]
                        ctr += 1
                    num_found += len(idx)
                    num_remaining = num_new - num_found


class KgeFrequencySampler(KgeSampler):
    """
    Sample negatives based on their relative occurrence in the slot in the train set.
    Can be smoothed with a symmetric prior.
    """

    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        self._multinomials = []
        alpha = self.get_option("frequency.smoothing")
        for slot in SLOTS:
            smoothed_counts = (
                np.bincount(
                    dataset.split(config.get("train.split"))[:, slot],
                    minlength=self.vocabulary_size[slot].item(),
                )
                + alpha
            )

            self._multinomials.append(
                torch._multinomial_alias_setup(
                    torch.from_numpy(smoothed_counts / np.sum(smoothed_counts))
                )
            )

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if num_samples == 0:
            result = torch.empty([positive_triples.size(0), num_samples])
        else:
            result = torch._multinomial_alias_draw(
                self._multinomials[slot][1],
                self._multinomials[slot][0],
                positive_triples.size(0) * num_samples,
            ).view(positive_triples.size(0), num_samples)

        return result
