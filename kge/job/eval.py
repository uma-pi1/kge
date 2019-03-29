import torch
from typing import List
import EntityRanking, EntityPairRanking


class EvalJob:

    def __init__(self, config, data, model=None):
        self.config = config
        self.dataset = data
        self.device = self.config.get('job.device')

        if not model:
            self.model = model
        else:
            # TODO load model from folder given in config file

    def create(config, dataset):
        """Factory method to create an evaluation job """
        if config.get('evaluation.type') == 'entity_ranking':
            return EntityRanking(config, dataset, model)
        elif config.get('evaluation.type') == 'entity_pair_ranking':
            return EntityPairRanking(config, dataset, model)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("evaluation.type")

    def compute_metrics(self, predictions, labels, filters) -> List[float]:
        """

        :param predictions:
        :param labels:
        :param filters:
        :return: list of metrics (floats)
        """

        raise NotImplementedError
