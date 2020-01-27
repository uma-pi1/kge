from kge import Config, Configurable, Dataset
from kge.indexing import where_in

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
                dataset.index(f"train_{pair}_to_{slot_str}")
        if any(self.filter_positives):
            self.check_option("filtering.implementation", ["standard", "fast"])
            self.filter_fast = True
            if self.get_option("filtering.implementation") == "standard":
                self.filter_fast = False
        self.dataset = dataset
        self.shared = self.get_option("shared")
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
            # TODO add frequency-based/biased sampling
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
            num_samples = self.num_samples[slot]
        if self.shared:
            negative_samples = self._sample_shared(
                positive_triples, slot, num_samples
            ).expand(positive_triples.size(0), num_samples)
        else:
            negative_samples = self._sample(positive_triples, slot, num_samples)
        if self.filter_positives[slot]:
            if self.shared:
                raise ValueError(
                    "Filtering is not supported when shared negative sampling is enabled."
                )
            negative_samples = self._filter(negative_samples, slot, positive_triples)
            if self.get_option("filtering.implementation") == "python":
                negative_samples = self._filter(negative_samples, slot, positive_triples)
            elif self.get_option("filtering.implementation") == "numba":
                negative_samples = self._filter_fast(negative_samples, slot, positive_triples)
            if self.filter_fast:
                negative_samples = self._filter_and_resample_fast(
                    negative_samples, slot, positive_triples
                )
            else:
                negative_samples = self._filter_and_resample(
                    negative_samples, slot, positive_triples
                )
        return negative_samples

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        """Sample negative examples."""
        raise NotImplementedError(
            "The selected sampler does not support shared negative samples."
        )

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        """Sample negative examples with shared entities for corruption.

        Returns a vector with with indexes of the sampled negative entities (`slot`=0 or
        `slot`=2) or relations (`slot`=1).
        """
        raise NotImplementedError()

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        """Filter and resample indices until only negatives have been created. """
        pair = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(f"train_{pair}_to_{SLOT_STR[slot]}")
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols]
        for i in range(positive_triples.size(0)):
            # indices of samples that have to be sampled again
            # Note: giving np.isin() a set as second argument potentially is faster
            # but conversion to a set induces costs that make things worse
            resample_idx = np.where(
                np.isin(negative_samples[i], index[tuple(pairs[i].tolist())]) != 0
            )[0]
            # number of new samples needed
            num_new = len(resample_idx)
            new = torch.empty(num_new, dtype=torch.long)
            # number already found of the new samples needed
            num_found = 0
            num_remaining = num_new - num_found
            while num_remaining:
                new_samples = self._sample(
                    positive_triples[i, None], slot, num_remaining
                ).flatten()
                # indices of the true negatives
                tn_idx = np.where(
                    np.isin(new_samples, index[tuple(pairs[i].tolist())]) == 0
                )[0]
                # store the correct (true negatives) samples found
                if len(tn_idx):
                    new[num_found : num_found + len(tn_idx)] = new_samples[tn_idx]
                num_found += len(tn_idx)
                num_remaining = num_new - num_found
            negative_samples[i, resample_idx] = new
        return negative_samples

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        """Filtering implementation for specific samplers.

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

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        pair = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(f"train_{pair}_to_{SLOT_STR[slot]}")
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols].numpy()
        batch_size = positive_triples.size(0)
        voc_size = self.vocabulary_size[slot]
        # filling a numba-dict here and then call the function was faster than 1. Using
        # numba lists 2. Using a python list and convert it to an np.array and use
        # offsets 3. Growing a np.array with np.append 4. leaving the loop in python and
        # calling a numba function within the loop
        positives = numba.typed.Dict
        for i in range(batch_size):
            positives[tuple(pairs[i].tolist())] = np.array(
                index[tuple(pairs[i].tolist())]
            )
        negative_samples = negative_samples.numpy()
        KgeUniformSampler._filter_and_resample_numba(
            positives, pairs, negative_samples, batch_size, int(voc_size),
        )
        return torch.tensor(negative_samples, dtype=torch.int64)

    @numba.njit
    def _filter_and_resample_numba(
        positives, pairs, negative_samples, batch_size, voc_size
    ):
        for i in range(batch_size):
            pos = positives[(pairs[i][0], pairs[i][1])]
            # inlining the idx_wherein function here results in an internal numba
            # error which asks to file a bug report
            resample_idx = where_in(negative_samples[i], pos, True)
            # number of new samples needed
            num_new = len(resample_idx)
            new = np.empty(num_new)
            # number already found of the new samples needed
            num_found = 0
            num_remaining = num_new - num_found
            while num_remaining:
                new_samples = np.random.randint(0, voc_size, num_remaining)
                idx = where_in(new_samples, pos, False)
                # store the correct (true negatives) samples found
                if len(idx):
                    new[num_found : num_found + len(idx)] = new_samples[idx]
                num_found += len(idx)
                num_remaining = num_new - num_found
            ctr = 0
            # numba does not support result[i, resample_idx] = new
            for j in resample_idx:
                negative_samples[i, j] = new[ctr]
                ctr += 1

    def get_numba_sampler(self):
        if isinstance(self, KgeUniformSampler):
            return KgeUniformSampler._sample_numba
        else:
            raise NotImplementedError(
                "No numba filtering implemented. Use filtering.implementation=python"
            )


class KgeUniformSampler(KgeSampler):
    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key, dataset)

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        return torch.randint(
            self.vocabulary_size[slot], (positive_triples.size(0), num_samples)
        )
    @njit
    def _sample_numba(voc_size, num_remaining):
        return np.random.randint(0, voc_size, num_remaining)

    def _sample_shared(
            self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        return self._sample(torch.empty(1), slot, num_samples).view(-1)


@njit
def idx_where_in(x, y, t_f=True):
    """ Retrieves the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: t_f: if False, returns the indices of the of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. This is faster than np.where(np.isin))
    # "i in y" raises an error in numba when y is a np.array.
    # casting y to a set instead a list was always slower in test scripts
    # setting njit(parallel=True) slowed down the function
    list_y = list(y)
    return np.where(np.array([i in list_y for i in x]) == t_f)[0]