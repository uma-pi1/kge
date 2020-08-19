from kge import Config, Configurable, Dataset
from kge.indexing import where_in

import random
import torch
from typing import Optional
import numpy as np
import numba

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
        self.with_replacement = self.get_option("with_replacement")
        if not self.with_replacement and not self.shared:
            raise ValueError(
                "Without random_replacement sampling is only supported when "
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
            self.check_option(
                "filtering.implementation", ["standard", "fast", "fast_if_available"]
            )

            self.filter_implementation = self.get_option("filtering.implementation")
        self.dataset = dataset
        # auto config
        for slot, copy_from in [(S, O), (P, None), (O, S)]:
            if self.num_samples[slot] < 0:
                if copy_from is not None and self.num_samples[copy_from] > 0:
                    self.num_samples[slot] = self.num_samples[copy_from]
                else:
                    self.num_samples[slot] = 0

    @staticmethod
    def create(config: Config, configuration_key: str, dataset: Dataset):
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
        shared_raw_results=False,
    ):
        """Obtain a set of negative samples for a specified slot.

        `positive_triples` is a batch_size x 3 tensor of positive triples. `slot` is
        either 0 (subject), 1 (predicate), or 2 (object). If `num_samples` is `None`,
        it is set to the default value for the slot configured in this sampler.

        Returns a batch_size x num_samples tensor with indexes of the sampled negative
        entities (`slot`=0 or `slot`=2) or relations (`slot`=1). When `shared_raw_results` is
        set, returns results as in `_sample_shared` instead.

        """
        if num_samples is None:
            num_samples = self.num_samples[slot].item()
        if self.shared:
            unique_samples, drop_index, repeat_indexes = self._sample_shared(
                positive_triples, slot, num_samples
            )

            if shared_raw_results:
                return unique_samples, drop_index, repeat_indexes
            else:
                # this will hold the result
                batch_size = len(positive_triples)
                negative_samples = torch.empty(
                    batch_size, num_samples, dtype=torch.long
                )

                # Add the first num_distinct samples for each positive. Dropping is
                # performed by copying the last shared sample over the dropped sample
                num_unique = len(unique_samples) - 1
                negative_samples[:, :num_unique] = unique_samples[:-1]
                drop_rows = torch.nonzero(
                    drop_index != num_unique, as_tuple=False
                ).squeeze()
                negative_samples[drop_rows, drop_index[drop_rows]] = unique_samples[-1]

                if num_unique != num_samples:
                    negative_samples[:, num_unique:] = negative_samples[
                        :, repeat_indexes
                    ]
        else:
            negative_samples = self._sample(positive_triples, slot, num_samples)
        if self.filter_positives[slot]:
            if self.filter_implementation == "fast":
                negative_samples = self._filter_and_resample_fast(
                    negative_samples, slot, positive_triples
                )
            elif self.filter_implementation == "standard":
                negative_samples = self._filter_and_resample(
                    negative_samples, slot, positive_triples
                )
            else:
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
        return negative_samples

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        """Sample negative examples."""
        raise NotImplementedError("The selected sampler is not implemented.")

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        """Sample negative examples with sharing.

        Returns:

        - a tensor `unique_samples` of size U+1 holding a list of unique negative
        samples. For each positive triple, U of these samples will be used.

        - a tensor `drop_index` that indicates for each positive triple, which unique
          sample is not used for that positive. The dropped sample should be replaced
          with the last entry in `unique_samples`. The option to drop a negative sample
          is used to avoid using the true positive from `positive_triples` as a negative
          sample: when that true positive is `unique_samples`, it should be ignored.

        - a tensor `repeat_indexes` of size `num_samples-U` holding the indexes of
          repeated unique samples

        """
        raise NotImplementedError(
            "The selected sampler does not support shared negative samples."
        )

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
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
    ):
        """Filter and resample indices.

        Samplers can override this method when their sampling strategy allows for a
        more efficient filtering method than the generic standard method or when their
        code can be optimized by tools such as Numba.

        """
        raise NotImplementedError(
            "Use filtering.implementation=standard for this sampler."
        )


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
        batch_size = len(positive_triples)

        # determine number of distinct negative samples for each positive
        if self.with_replacement:
            # Simple way to get a sample from the distribution of number of distinct
            # values in the negative sample (WR sampling except the positive, hence the
            # -1)
            num_unique = len(
                np.unique(
                    np.random.choice(
                        self.vocabulary_size[slot] - 1, num_samples, replace=True
                    )
                )
            )
        else:  # WOR -> all distinct
            num_unique = num_samples

        # Take one more WOR sample than necessary (used to replace sampled positives).
        # Numpy is horribly slow for large vocabulary sizes, so we use random.sample
        # instead
        #
        # unique_samples = np.random.choice(
        #     self.vocabulary_size[slot], num_unique + 1, replace=False
        # )
        unique_samples = random.sample(range(self.vocabulary_size[slot]), num_unique + 1)

        # For each row i (positive triple), select a sample to drop. For rows that
        # contain its positive, drop that positive. For all other rows, drop a random
        # position. Here we start with random position for each row:
        drop_index = np.random.choice(num_unique + 1, batch_size, replace=True)
        # and then update the ones that contain its positive in the negative samples
        positives = positive_triples[:, slot].numpy()
        unique_samples_index = {s: j for j, s in enumerate(unique_samples)}
        for i, v in [
            (i, unique_samples_index.get(positives[i]))
            for i in range(batch_size)
            if positives[i] in unique_samples_index
        ]:
            drop_index[i] = v

        # For WOR, we are done (tensor will be []). For WR, we need to upsample. To do
        # so, we compute the set of additional (repeated) sample indexes.
        if num_unique != num_samples:  # only happens with WR
            repeat_indexes = torch.tensor(
                np.random.choice(num_unique, num_samples - num_unique, replace=True)
            )
        else:
            repeat_indexes = torch.empty(0)

        return torch.tensor(unique_samples), torch.tensor(drop_index), repeat_indexes

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        pair_str = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(
            f"{self.filtering_split}_{pair_str}_to_{SLOT_STR[slot]}"
        )
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols].numpy()
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
