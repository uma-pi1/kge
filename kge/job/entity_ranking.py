import math
import time

import torch
import kge.job
from kge.job import EvaluationJob


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = False

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        # create indexes
        self.train_sp = self.dataset.index_1toN("train", "sp")
        self.train_po = self.dataset.index_1toN("train", "po")
        self.valid_sp = self.dataset.index_1toN("valid", "sp")
        self.valid_po = self.dataset.index_1toN("valid", "po")

        if self.eval_data == "test":
            self.triples = self.dataset.test
        else:
            self.triples = self.dataset.valid
        if self.eval_data == "test" or self.filter_valid_with_test:
            self.test_sp = self.dataset.index_1toN("test", "sp")
            self.test_po = self.dataset.index_1toN("test", "po")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        train_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
            batch, self.dataset.num_entities, self.train_sp, self.train_po
        )
        valid_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
            batch, self.dataset.num_entities, self.valid_sp, self.valid_po
        )
        if self.eval_data == "test" or self.filter_valid_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch, self.dataset.num_entities, self.test_sp, self.test_po
            )
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        batch = torch.cat(batch).reshape((-1, 3))
        return batch, train_label_coords, valid_label_coords, test_label_coords

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on " + self.eval_data + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities

        # we also filter with test data during validation if requested
        filtered_valid_with_test = (
            self.eval_data == "valid" and self.filter_valid_with_test
        )

        # Initiliaze dictionaries that hold the overall histogram of ranks, which are
        # used to compute the metrics. The dictionary entry with key 'all' collects
        # the overall statistics which is added by default into self.hist_hooks.
        hist_dict = dict()
        hist_filt_dict = dict()
        hist_filt_test_dict = dict()

        for f in self.hist_hooks:
            hist_dict.update(f.init_hist_hook(self, self.dataset, device=self.device, dtype=torch.float))
            hist_filt_dict.update(f.init_hist_hook(self, self.dataset, device=self.device, dtype=torch.float))
            hist_filt_test_dict.update(f.init_hist_hook(self, self.dataset, device=self.device, dtype=torch.float))

        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # construct a label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            train_label_coords = batch_coords[1].to(self.device)
            valid_label_coords = batch_coords[2].to(self.device)
            test_label_coords = batch_coords[3].to(self.device)
            if self.eval_data == "test":
                label_coords = torch.cat(
                    [train_label_coords, valid_label_coords, test_label_coords]
                )
            else:  # it's valid
                label_coords = torch.cat([train_label_coords, valid_label_coords])
                if filtered_valid_with_test:
                    test_labels = kge.job.util.coord_to_sparse_tensor(
                        len(batch),
                        2 * num_entities,
                        test_label_coords,
                        self.device,
                        float("Inf"),
                    ).to_dense()
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            ).to_dense()

            # compute all scores
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            scores = self.model.score_sp_po(s, p, o)
            scores_sp = scores[:, :num_entities]
            scores_po = scores[:, num_entities:]

            # compute raw ranks
            s_ranks, o_ranks, _, _ = self._filter_and_rank(
                s, p, o, scores_sp, scores_po, None
            )
            # same for filtered ranks
            s_ranks_filt, o_ranks_filt, scores_sp_filt, scores_po_filt = self._filter_and_rank(
                s, p, o, scores_sp, scores_po, labels
            )

            # Now compute the batch histogram of raw ranks and potentially mask
            # out batch items (computed by mask_out_batch_ranks_hook) and
            # aggregate the histograms
            for f in self.hist_hooks:
                for key, mask in f.mask_out_batch_ranks_hook(self, self.dataset, s, p, o):
                    hist_dict[key] += self._make_batch_hist(mask, s_ranks, o_ranks)
                    hist_filt_dict[key] += self._make_batch_hist(mask, s_ranks_filt, o_ranks_filt)

            # and the same for filtered_with_test ranks
            if filtered_valid_with_test:
                s_ranks_filt_test, o_ranks_filt_test, _, _ = self._filter_and_rank(
                    s, p, o, scores_sp_filt, scores_po_filt, test_labels
                )
                for f in self.hist_hooks:
                    for key, mask in f.mask_out_batch_ranks_hook(self, self.dataset, s, p, o):
                        hist_filt_test_dict[key] += self._make_batch_hist(mask, s_ranks_filt_test, o_ranks_filt_test)

            # optionally: trace ranks of each example
            if self.trace_examples:
                entry = {
                    "type": "entity_ranking",
                    "scope": "example",
                    "data": self.eval_data,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                    )
                    if filtered_valid_with_test:
                        entry["rank_filtered_with_test"] = (
                            o_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry,
                    )
                    if filtered_valid_with_test:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )

            # now compute the batch metrics
            metrics = self._compute_metrics(hist_dict['all'])
            metrics.update(self._compute_metrics(hist_filt_dict['all'], suffix="_filtered"))
            if filtered_valid_with_test:
                metrics.update(
                    self._compute_metrics(
                        hist_filt_test_dict['all'], suffix="_filtered_with_test"
                    )
                )

            # TODO: changed semantics of contents of batch metrics. Before: metric per batch, Now: metric until current batch, makes for nicer printing but not traceable
            # # optionally: trace batch metrics
            # if self.trace_batch:
            #     self.trace(
            #         type="entity_ranking",
            #         scope="batch",
            #         data=self.eval_data,
            #         epoch=self.epoch,
            #         batch=batch_number,
            #         size=len(batch),
            #         batches=len(self.loader),
            #         **metrics,
            #     )

            # output batch information to console
            print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

        # we are done; compute final metrics
        print("\033[2K\r", end="", flush=True)  # clear line and go back

        metrics = dict()

        for hist_name in hist_dict.keys():

            if hist_name == 'all':
                hist_print_name = ''
            else:
                hist_print_name = '_' + hist_name

            metrics.update(self._compute_metrics(hist_dict[hist_name], suffix=hist_print_name))
            metrics.update(self._compute_metrics(hist_filt_dict[hist_name], suffix="_filtered" + hist_print_name))

        if filtered_valid_with_test:
            for hist_name in hist_dict.keys():
                if hist_name == 'all':
                    hist_print_name = ''
                else:
                    hist_print_name = '_' + hist_name
                metrics.update(
                    self._compute_metrics(hist_filt_test_dict[hist_name], suffix="_filtered_with_test" + hist_print_name)
                )
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="entity_ranking",
            scope="epoch",
            data=self.eval_data,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
            epoch_time=epoch_time,
            **metrics,
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **trace_entry),
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_data + " data.")

        return trace_entry

    def _filter_and_rank(self, s, p, o, scores_sp, scores_po, labels):
        num_entities = self.dataset.num_entities
        if labels is not None:
            # remove current example from labels
            indices = torch.arange(0, len(o)).long()
            labels[indices, o.long()] = 0
            labels[indices, (s + num_entities).long()] = 0
            labels_sp = labels[:, :num_entities]
            labels_po = labels[:, num_entities:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_ranks = self._get_rank(scores_sp, o)
        s_ranks = self._get_rank(scores_po, s)
        return s_ranks, o_ranks, scores_sp, scores_po

    def _make_batch_hist(self, mask, o_ranks, s_ranks):
        num_entities = self.dataset.num_entities
        batch_hist = torch.zeros([num_entities], device=self.device, dtype=torch.float)
        # need for loop because batch_hist[o_ranks]+=1 ignores repeated
        # entries in o_ranks
        for r, m in zip(o_ranks, mask):
            if m: batch_hist[r] += 1
        for r, m in zip(s_ranks, mask):
            if m: batch_hist[r] += 1
        return batch_hist

    def _get_rank(self, scores, answers):
        # Get scores of answer given by each triple
        # Add small number to all scores to make the rank of the true worst in case of tie
        # Get tensor of 1s for each score which is higher than the true answer score.
        # Add 1s in each row to get the rank of the corresponding row.
        # Fix for the add small number fix we substract 1 from each rank with
        # the lowest possible.
        true_scores = scores[range(answers.size(0)), answers.long()]
        scores = scores + 1e-40
        ranks = torch.sum((scores > true_scores.view(-1, 1)).long(), dim=1)
        ranks = ranks - (ranks == self.dataset.num_entities).long()
        return ranks

    def _compute_metrics(self, rank_hist, suffix=""):
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (torch.sum(rank_hist * ranks).item() / n) if n > 0. else 0.

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            torch.sum(rank_hist * reciprocal_ranks).item() / n
        ) if n > 0. else 0.

        hits_at_k = (
            torch.cumsum(rank_hist[: max(self.hits_at_k_s)], dim=0) / n
        ).tolist() if n > 0. else [0.] * max(self.hits_at_k_s)

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics
