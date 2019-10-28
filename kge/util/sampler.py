import torch
import random

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


class TripleClassificationSampler(KgeNegativeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)

    def sample(self, dataset):
        """Generates dataset with positive and negative triples.

        Takes each triple of the specified dataset and randomly replaces either the subject or the object with another
        subject/object. Only allows a subject/object to be sampled if it appeared as a subject/object at the same
        position in the dataset.

        Returns:
            corrupted: A new dataset with the original and corrupted triples.

            labels: A vector with labels for the corresponding triples in the dataset.

            rel_labels: A dictionary mapping relations to labels.
                        Example if we had two triples of relation 1 in the original dataset: {1: [1, 0, 1, 0]}
        """

        # Create objects for the corrupted dataset and the corresponding labels
        corrupted = dataset.repeat(1, 2).view(-1, 3)
        labels = torch.as_tensor([1, 0] * len(dataset)).to(self.device)

        # The sampling influences the results in the end. To compare models or parameters, the seeds should be fixed
        if self.config.get("eval.triple_classification_random_seed"):
            torch.manual_seed(5465456876546785)
            random.seed(5465456876546785)

        # Random decision if sample subject(sample=nonzero) or object(sample=zero)
        sample = torch.randint(0, 2, (1, len(dataset))).to(self.device)

        # Sample subjects from subjects which appeared in the dataset
        corrupted[1::2][:, 0][sample.nonzero()[:, 1]] = \
            torch.as_tensor(random.choice(
                list(map(int, list(map(int, dataset[:, 0].unique()))))), dtype=torch.int32).to(self.device)

        # Sample objects from objects which appeared in the dataset
        corrupted[1::2][:, 2][(sample == 0).nonzero()[:, 1]] = \
            torch.as_tensor(random.choice(
                list(map(int, list(map(int, dataset[:, 2].unique()))))), dtype=torch.int32).to(self.device)

        # Save the labels per relation, since this will be needed frequently later on
        p = corrupted[:, 1]
        rel_labels = {int(r): labels[p == r] for r in p.unique()}

        return corrupted, labels, rel_labels
