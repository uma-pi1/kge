import torch
from kge.evaluation import EvaluationJob


class EntityRanking(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, data, model):
        super().__init__(config, data, model)

        self.loader = torch.utils.data.DataLoader(data,
                                                  collate_fn=self._collate,
                                                  shuffle=False,
                                                  batch_size=self.batch_size,
                                                  num_workers=config.get('train.num_workers'),
                                                  pin_memory=config.get('train.pin_memory'))
