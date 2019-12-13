from kge import Configurable
import torch
import numpy as np

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class KgeNegativeSampler(Configurable):
    """ Negative sampler """

    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key)

        self.num_negatives = dict()
        self._filter_true = dict()
        self.voc_size = dict()

        for slot, num_negatives_key, filter_true_key, voc_size in [
            (S, "num_negatives_s", "filter_true_s", dataset.num_entities),
            (P, "num_negatives_p", "filter_true_p", dataset.num_relations),
            (O, "num_negatives_o", "filter_true_o", dataset.num_entities),
        ]:
            self.num_negatives[slot] = self.get_option(num_negatives_key)
            self._filter_true[slot] = self.get_option(filter_true_key)
            self.voc_size[slot] = voc_size

        for slot, copy_from in [(S, O), (P, None), (O, S)]:
            if self.num_negatives[slot] < 0:
                if copy_from is not None and self.num_negatives[copy_from] > 0:
                    self.num_negatives[slot] = self.num_negatives[copy_from]
                else:
                    self.num_negatives[slot] = 0

        self.num_negatives_total = sum(self.num_negatives.values())

    @staticmethod
    def create(config, configuration_key, dataset=None):
        """ Factory method for sampler creation """
        sampling_type = config.get(configuration_key + ".sampling_type")
        if sampling_type == "uniform":
            return UniformSampler(config, configuration_key, dataset)
        elif sampling_type == "frequency_based":
            return FrequencyBasedSampler(config, configuration_key, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(configuration_key + ".sampling_type")

    def sample(self, spo, slot, num_samples=None):
        raise NotImplementedError()

    def _filter(self, result):
        raise NotImplementedError()


class UniformSampler(KgeNegativeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)

    def sample(self, spo, slot, num_samples=None):
        if num_samples is None:
            num_samples = self.num_negatives[slot]
        result = torch.randint(self.voc_size[slot], (spo.size(0), num_samples))
        if self._filter_true[slot]:
            result = self._filter(result)
        return result


class FrequencyBasedSampler(KgeNegativeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        self.distributions = []
        for slot in SLOTS:
            # NOTE: we have to add a probability of 0 for the non occurring entities / relations here
            # entities could for example only occur as subject but not as object
            vocab = np.arange(self.voc_size[slot])
            unique, counts = np.unique(dataset.train[:, slot], return_counts=True)
            missing = vocab[~np.isin(vocab, unique)]
            unique = np.concatenate([unique, missing])
            probabilities = np.concatenate([counts, np.zeros(len(missing))])/np.sum(counts)
            sort_index = np.argsort(unique)
            self.distributions.append(probabilities[sort_index])

    def sample(self, spo, slot, num_samples=None):
        if num_samples is None:
            num_samples = self.num_negatives[slot]
        result = np.random.choice(self.voc_size[slot], spo.size(0)*num_samples, p=self.distributions[slot])
        result = torch.from_numpy(result).view([spo.size(0), num_samples])
        if self._filter_true[slot]:
            result = self._filter(result)
        return result
