import math
import torch
import kge.job
from kge.job import EvaluationJob


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, dataset, model, what='test'):
        super().__init__(config, dataset, model)

        if what != 'test' and what != 'valid':
            raise ValueError('what')
        self.what = what

        # create indexes
        self.train_sp = dataset.index_1toN('train', 'sp')
        self.train_po = dataset.index_1toN('train', 'po')
        self.valid_sp = dataset.index_1toN('valid', 'sp')
        self.valid_po = dataset.index_1toN('valid', 'po')
        self.triples = dataset.valid
        if what == 'test':
            self.triples = dataset.test
            self.test_sp = dataset.index_1toN('test', 'sp')
            self.test_po = dataset.index_1toN('test', 'po')

        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=config.get('eval.num_workers'),
            pin_memory=config.get('eval.pin_memory'))

    def _collate(self, batch):
        train_label_coords = kge.job.util.get_batch_sp_po_coords(
            batch, self.dataset.num_entities, self.train_sp, self.train_po)
        valid_label_coords = kge.job.util.get_batch_sp_po_coords(
            batch, self.dataset.num_entities, self.valid_sp, self.valid_po)
        if self.what == 'test':
            test_label_coords = kge.job.util.get_batch_sp_po_coords(
                batch, self.dataset.num_entities, self.test_sp, self.test_po)
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        batch = torch.cat(batch).reshape((-1, 3))
        return batch, train_label_coords, valid_label_coords, test_label_coords

    # TODO devices! All on selected device? Better if something on CPU?
    def run(self) -> dict:
        was_training = self.model.training
        self.model.eval()
        self.config.log("Evaluating " + self.what + " data (epoch {})...".format(self.epoch))
        num_entities = self.dataset.num_entities
        hist = torch.zeros([num_entities], device=self.device, dtype=torch.float)
        hist_filtered = torch.zeros([num_entities], device=self.device, dtype=torch.float)
        for batch_number, batch_coords in enumerate(self.loader):
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            train_label_coords = batch_coords[1].to(self.device)
            valid_label_coords = batch_coords[2].to(self.device)
            test_label_coords = batch_coords[3].to(self.device)
            label_coords = torch.cat([train_label_coords, valid_label_coords, test_label_coords])
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2*num_entities, label_coords, self.device, float('Inf')).to_dense()

            # get scores
            s = batch[:, 0]
            p = batch[:, 1]
            o = batch[:, 2]
            scores = self.model.score_sp_po(s, p, o)
            scores_sp = scores[:, :num_entities]
            scores_po = scores[:, num_entities:]

            # compute raw ranks rank
            o_ranks = self._get_rank(scores_sp, o)
            s_ranks = self._get_rank(scores_po, s)

            # now filter
            # TODO this should be doable much more efficiently than by
            # constructing the label tensor (with Inf entries for seen entites)
            for i in range(len(batch)):  # remove current example from labels
                labels[i, o[i]] = 0
                labels[i, num_entities+s[i]] = 0
            labels_sp = labels[:, :num_entities]
            labels_po = labels[:, num_entities:]
            scores_sp_filtered = scores_sp - labels_sp
            scores_po_filtered = scores_po - labels_po

            # compute filtered ranks
            o_ranks_filtered = self._get_rank(scores_sp_filtered, o)
            s_ranks_filtered = self._get_rank(scores_po_filtered, s)

            # output ranks of each triple
            if self.config.get('eval.trace_examples'):
                for i in range(len(batch)):
                    self.config.trace(
                        type='eval_er', scope='example', data=self.what,
                        epoch=self.epoch,
                        batch=i, size=len(batch), batches=len(self.loader),
                        s=s[i].item(), p=p[i].item(), o=o[i].item(), task='sp',
                        rank=o_ranks[i].item()+1,
                        rank_filtered=o_ranks_filtered[i].item()+1)
                    self.config.trace(
                        type='eval_er', scope='example', data=self.what,
                        epoch=self.epoch,
                        batch=i, size=len(batch), batches=len(self.loader),
                        s=s[i].item(), p=p[i].item(), o=o[i].item(), task='po',
                        rank=s_ranks[i].item()+1,
                        rank_filtered=s_ranks_filtered[i].item()+1)

            # compute histogram of ranks
            batch_hist = torch.zeros([num_entities], device=self.device, dtype=torch.float)
            batch_hist_filtered = torch.zeros([num_entities], device=self.device, dtype=torch.float)
            for r in o_ranks: batch_hist[r] += 1
            for r in s_ranks: batch_hist[r] += 1
            for r in o_ranks_filtered: batch_hist_filtered[r] += 1
            for r in s_ranks_filtered: batch_hist_filtered[r] += 1
            hist += batch_hist
            hist_filtered += batch_hist_filtered

            # now get the metrics
            metrics = self._get_metrics(batch_hist)
            metrics.update(self._get_metrics(batch_hist_filtered, suffix='_filtered'))
            self.config.trace(type='eval_er', scope='batch', data=self.what,
                              epoch=self.epoch,
                              batch=i, size=len(batch), batches=len(self.loader),
                              **metrics)

            # output information
            print('\033[K\r', end="")  # clear line and go back
            print(('  batch:{: '
                   + str(1+int(math.ceil(math.log10(len(self.loader)))))
                   + 'd}/{}, mrr (filtered): {:5.4f} ({:5.4f})')
                  .format(batch_number, len(self.loader)-1,
                          metrics['mean_reciprocal_rank'],
                          metrics['mean_reciprocal_rank_filtered']),
                  end='')

        print("\033[2K\r", end="")  # clear line and go back
        metrics = self._get_metrics(hist)
        metrics.update(self._get_metrics(hist_filtered, suffix='_filtered'))
        self.config.trace(
            echo=True, echo_prefix="  ", log=True,
            type='eval_er', scope='epoch', data=self.what,
            epoch=self.epoch, batches=len(self.loader),
            size=len(self.triples),
            **metrics)

        if was_training: self.model.train()
        self.config.log("Finished evaluating " + self.what + " data.")
        return metrics

    def _get_rank(self, scores, answers):
        answers = answers.reshape((-1, 1)).expand(-1, self.dataset.num_entities).long()
        true_scores = torch.gather(scores, 1, answers)
        ranks = torch.sum((scores > true_scores).long(), dim=1)
        return ranks

    def _get_metrics(self, rank_hist, suffix=''):
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.tensor(range(self.dataset.num_entities), device=self.device).float() + 1.0
        metrics["mean_rank" + suffix] = torch.sum(rank_hist * ranks).item()/n

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = torch.sum(rank_hist * reciprocal_ranks).item()/n

        # TODO hits@k -> field: "hits_at_k: [ array with k elements ]", use prefix sum (cumsum)

        return metrics
