import torch


class KgeNegativeSampler:
    """ Negative sampler """

    def __init__(self, config, configuration_key, dataset=None, ):

        self.dataset = dataset

        # if num_s < 0 set num_s to num_o
        self._num_negatives_s = config.get(configuration_key + ".num_negatives_s")
        self._num_negatives_o = config.get(configuration_key + ".num_negatives_o")

        if self._num_negatives_s < 0:
            if self._num_negatives_o > 0:
                self._num_negatives_s = self._num_negatives_o
            else:
                self._num_negatives_s = 0

        # if num_o < 0 set num_o to num_s
        if self._num_negatives_o < 0:
            if self._num_negatives_s > 0:
                self._num_negatives_o = self._num_negatives_s
            else:
                self._num_negatives_o = 0

        if configuration_key in ['negative_sampling_spo']:
            self._num_negatives_p = config.get(configuration_key + ".num_negatives_p")
            if self._num_negatives_p < 0:
                self._num_negatives_p = 0

        if configuration_key in ['negative_sampling_spo']:
            self.num_negatives = self._num_negatives_s + \
                                 self._num_negatives_p + \
                                 self._num_negatives_o
        elif configuration_key in ['negative_sampling']:
            self.num_negatives = self._num_negatives_s + \
                                 self._num_negatives_o

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

    def sample(self, tuple_, type_:str):
        raise NotImplementedError()


class UniformSampler(KgeNegativeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)

    def sample(self, num_entities, num_negatives):
        return torch.randint( num_entities, (num_negatives, ))

