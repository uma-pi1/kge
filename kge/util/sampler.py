import torch


class KgeSampler:
    """ Negative sampler """

    def __init__(self, config, dataset):
        self.dataset = dataset
        self._num_negatives_s = config.get("negative_sampling.num_negatives_s")
        self._num_negatives_p = config.get("negative_sampling.num_negatives_p")
        self._num_negatives_o = config.get("negative_sampling.num_negatives_o")

        # Assumes num_negatives_s cannot be -1
        if self._num_negatives_s < 0:
            raise ValueError("num_negatives_s cannot be -1")
        # TODO Not supported for now, margin ranking assumes same for both s and o
        if self._num_negatives_o >= 0 and self._num_negatives_o != self._num_negatives_s:
            raise ValueError("num_negatives_o different from num_negatives_s not yet supported.")
        # TODO Add support for sampling relations
        # Issue is dataset is sp and po tuples, which ones will have p replaced
        if self._num_negatives_p > 0:
            raise ValueError("num_negatives_p, sampling relations not yet supported")

        # TODO For now, these stay the same
        self._num_negatives_o = self._num_negatives_s

    @staticmethod
    def create(config, dataset):
        """ Factory method for sampler creation """
        sampling_type = config.get("negative_sampling.sampling_type")
        if sampling_type == "uniform":
            return UniformSampler(config, dataset)
        # TODO add frequency-based/biased sampling
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("negative_sampling.sampling_type")

    def __call__(self, tuple_, type_:str):
        return self._sample(tuple_, type_)

    def _sample(self, tuple_, type_:str):
        raise NotImplementedError()


class UniformSampler(KgeSampler):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._filter_with_training_set = config.get("negative_sampling.filter_negatives")
        if self._filter_with_training_set:
            self._train_sp = self.dataset.index_1toN("train", "sp")
            self._train_po = self.dataset.index_1toN("train", "po")

    def _sample(self, tuple_, type_:str):
        """Generates negative candidates for given tuple and type (sp or po)"""

        if type_ == "sp":
            random_entities = torch.randint(
                0, self.dataset.num_entities, (self._num_negatives_o, 1)
            )
        elif type_ == "po":
            random_entities = torch.randint(
                0, self.dataset.num_entities, (self._num_negatives_s, 1)
            )
        else:
            raise Exception("Unrecognized type of tuple in sample function")

        if not self._filter_with_training_set:
            return random_entities
        else:
            # Filter out triples in training
            filtered_random_entities = []
            if type_ == "sp":
                index = self._train_sp
            elif type_ == "po":
                index = self._train_po
            else:
                raise Exception("Unrecognized type of tuple in sample function")
            for entity in random_entities:
                while entity in index[tuple_]:
                    entity = torch.randint(
                        0, self.dataset.num_entities, (1, 1)
                    ).item()
                filtered_random_entities.append(entity.item())
            return torch.tensor(filtered_random_entities)
