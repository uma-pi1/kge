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
    ):
        """Obtain a set of negative samples for a specified slot.

        `positive_triples` is a batch_size x 3 tensor of positive triples. `slot` is
        either 0 (subject), 1 (predicate), or 2 (object). If `num_samples` is `None`,
        it is set to the default value for the slot configured in this sampler.

        Returns a batch_size x num_samples tensor with indexes of the sampled negative
        entities (`slot`=0 or `slot`=2) or relations (`slot`=1).

        """
        if num_samples is None:
            num_samples = self.num_samples[slot].item()
        if self.shared:
            negative_samples = self._sample_shared(positive_triples, slot, num_samples)
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
        raise NotImplementedError(
            "The selected sampler is not implemented."
        )

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        """Sample negative examples with sharing.

        The negative samples returned by this method are shared for the positive triples
        to the amount possible.

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
            # Crude way to get the distribution of number of distinct values in the
            # negative sample (WR sampling except the positive, hence the -1)
            num_distinct = len(
                np.unique(
                    np.random.choice(
                        self.vocabulary_size[slot] - 1, num_samples, replace=True
                    )
                )
            )
        else:  # WOR -> all distinct
            num_distinct = num_samples

        # Take one more WOR sample than necessary (used to replace sampled positives).
        # Numpy is horribly slow for large vocabulary sizes, so we use random.sample
        # instead
        #
        # shared_samples = np.random.choice(
        #     self.vocabulary_size[slot], num_distinct + 1, replace=False
        # )
        shared_samples = random.sample(
            range(self.vocabulary_size[slot]), num_distinct + 1
        )

        # For each row i (positive triple), select a sample to drop. For rows that
        # contain its positive, drop that positive. For all other rows, drop a random
        # position.
        shared_samples_index = {s: j for j, s in enumerate(shared_samples)}
        replacement = np.random.choice(
            num_distinct + 1, batch_size, replace=True
        )
        drop = torch.tensor(
            [
                shared_samples_index.get(s, replacement[i])
                for i, s in enumerate(positive_triples[:, slot].numpy())
            ]
        )

        # this will hold the result
        samples = torch.empty(batch_size, num_samples, dtype=torch.long)

        # Add the first num_distinct samples for each positive. Dropping is performed by
        # copying the last shared sample over the dropped sample
        samples[:, :num_distinct] = torch.tensor(shared_samples[:-1])
        update_rows = torch.nonzero(drop != num_distinct).squeeze()
        samples[update_rows, drop[update_rows]] = shared_samples[-1]

        # samples now contains num_distinct WOR samples per triple and no positive. For
        # WOR, we are done. For WR, upsample
        if num_distinct != num_samples:  # only happens with WR
            indexes = torch.tensor(
                np.random.choice(num_distinct, num_samples - num_distinct, replace=True)
            )
            samples[:, num_distinct:] = samples[:, indexes]

        return samples

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
            positives_index[pair] = np.array(index.get(pair), dtype=np.int32)
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
