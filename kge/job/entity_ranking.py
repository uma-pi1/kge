import torch
import numpy
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
        # batch is list of tensors, one tensor for each triple
        train_filter = []
        valid_filter = []
        test_filter = []

        batch = torch.cat(batch).reshape((-1,3))
        return batch, train_filter, valid_filter, test_filter

    # TODO devices! All on selected device? Better if something on CPU?
    def run(self):
        num_entities = self.dataset.num_entities
        for i, batch_and_filters in enumerate(self.loader):
            batch = batch_and_filters[0].to(self.device)
            # train_filter = batch_and_filters[1].to(self.device)
            # valid_filter = batch_and_filters[2].to(self.device)
            # test_filter = batch_and_filters[3].to(self.device)

            # Get scores
            s = batch[:, 0]
            p = batch[:, 1]
            o = batch[:, 2]
            scores = self.model.score_sp_po(s, p, o)
            scores_sp = scores[:, :num_entities]
            scores_po = scores[:, num_entities:]

            # TODO filtering

            # Sort scores TODO quick select first better?
            sorted_sp = torch.argsort(scores_sp, dim=1, descending=True)
            sorted_po = torch.argsort(scores_po, dim=1, descending=True)

            # Get position of correct answer
            answers_o = o.reshape((-1, 1)).expand(-1, num_entities).long()
            answers_s = s.reshape((-1, 1)).expand(-1, num_entities).long()
            positions_sp = torch.argmax(sorted_sp == answers_o, dim=1)
            positions_po = torch.argmax(sorted_po == answers_s, dim=1)

            # Compute metrics
            self._compute_metrics(positions_sp)
            self._compute_metrics(positions_po)

            # TODO output to trace file (single line)

    def _get_rank(self, scores, answer):
        return np.where(scores == answer, scores, 0)

    def _compute_metrics(self, ranks):
        # TODO compute hits at lower values of k, e.g. 1, 3
        hits_at_k = torch.where(ranks < self.k, torch.ones(ranks.shape), torch.zeros(ranks.shape)).sum().item()
        # TODO fix! 1/ranks returns all zeros, even though docs say that is the way to compute elementwise recip
        mrr = torch.mean((1/ranks).float()).item()

        return hits_at_k, mrr
