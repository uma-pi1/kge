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
        triples = dataset.valid
        if what == 'test':
            triples = dataset.test
            self.test_sp = dataset.index_1toN('test', 'sp')
            self.test_po = dataset.index_1toN('test', 'po')

        self.loader = torch.utils.data.DataLoader(
            triples,
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
    def run(self):
        self.model.eval()
        self.config.log("Evaluating " + self.what + "...")
        num_entities = self.dataset.num_entities
        for batch_number, batch_coords in enumerate(self.loader):
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            train_label_coords = batch_coords[1].to(self.device)
            valid_label_coords = batch_coords[2].to(self.device)
            test_label_coords = batch_coords[3].to(self.device)
            label_coords = torch.cat([train_label_coords, valid_label_coords, test_label_coords])
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2*self.dataset.num_entities, label_coords, self.device, float('Inf')).to_dense()

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
                        type='eval_er_example' if self.what == 'test' else 'valid',
                        epoch=self.epoch,
                        batch=i, size=len(batch), batches=len(self.loader),
                        s=s[i].item(), p=p[i].item(), o=o[i].item(), task='sp',
                        rank=o_ranks[i].item()+1,
                        filtered_rank=o_ranks_filtered[i].item()+1)
                    self.config.trace(
                        type='eval_er_example' if self.what == 'test' else 'valid',
                        epoch=self.epoch,
                        batch=i, size=len(batch), batches=len(self.loader),
                        s=s[i].item(), p=p[i].item(), o=o[i].item(), task='po',
                        rank=s_ranks[i].item()+1,
                        filtered_rank=s_ranks_filtered[i].item()+1)

            # output information
            print('\033[K\r', end="")  # clear line and go back
            print(('  batch:{: '
                   + str(1+int(math.ceil(math.log10(len(self.loader)))))
                   + 'd}/{}').format(batch_number, len(self.loader)-1), end='')


            #     n += 1
            # now crfeate histogram of ranks for batch -> trace MRR/HITS from that

            # add to histogram of entire dataset -> trace MRR HITS after for loop

            # Compute metrics
            self._compute_metrics(o_ranks)
            self._compute_metrics(s_ranks)

            # TODO output to trace file (single line)

        print("\033[2K\r", end="")  # clear line and go back

        self.config.log("Finished evaluating " + self.what + "...")

    def _get_rank(self, scores, answers):
        order = torch.argsort(scores, dim=1, descending=True)
        answers = answers.reshape((-1, 1)).expand(-1, self.dataset.num_entities).long()
        ranks = torch.argmax(order == answers, dim=1)  # position of correct answer
        return ranks

    def _compute_metrics(self, ranks):
        # TODO compute hits at lower values of k, e.g. 1, 3
        # hits_at_k = torch.where(ranks < self.max_k, torch.ones(ranks.shape), torch.zeros(ranks.shape)).sum().item()
        # TODO fix! 1/ranks returns all zeros, even though docs say that is the way to compute elementwise recip
        # mrr = torch.mean((1/ranks).float()).item()

        return 0,0 # hits_at_k, mrr
