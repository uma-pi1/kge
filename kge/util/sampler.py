import torch

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class KgeNegativeSampler:
    """ Negative sampler """

    def __init__(self, config, configuration_key, dataset):

        self.num_negatives = dict()

        self._filter_true = dict()
        self.voc_size = dict()

        for slot, num_negatives_key, filter_true_key, voc_size in [
            (S, ".num_negatives_s", ".filter_true_s", dataset.num_entities),
            (P, ".num_negatives_p", ".filter_true_p", dataset.num_relations),
            (O, ".num_negatives_o", ".filter_true_o", dataset.num_entities),
        ]:
            self.num_negatives[slot] = config.get(configuration_key + num_negatives_key)
            self._filter_true[slot] = config.get(configuration_key + filter_true_key)
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
            # TODO add frequency-based/biased sampling
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
