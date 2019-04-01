import torch
from kge.job import EvalJob


class EntityRanking(EvalJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)

        # create dataloader
        if config.get('job.type') == 'train':
            data = dataset.valid
        else:
            data = dataset.test
        self.loader = torch.utils.data.DataLoader(data,
                                                  collate_fn=self._collate,
                                                  shuffle=False,
                                                  batch_size=self.batch_size,
                                                  num_workers=config.get('train.num_workers'),
                                                  pin_memory=config.get('train.pin_memory'))

    def _collate(self, batch):
        # TODO returns batch and filters for train, valid, test data
        pass

    # TODO devices! All on selected device? Better if something on CPU?
    def run(self):
        for i, batch_and_filters in enumerate(self.loader):
            batch = batch_and_filters[0].to(self.device)
            train_filter = batch_and_filters[1].to(self.device)
            valid_filter = batch_and_filters[2].to(self.device)
            test_filter = batch_and_filters[3].to(self.device)

            # Get scores
            scores = self.model.score_sp_po(batch[:, 0], batch[:, 1], batch[:, 2])

            # TODO compute predictions and metrics

            # TODO output to trace file (single line)


