import math
import time

import torch
import kge.job
from kge.job import EvaluationJob, Job


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = False

        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        # create indexes
        self.train_sp = self.dataset.index_KvsAll("train", "sp")
        self.train_po = self.dataset.index_KvsAll("train", "po")
        self.valid_sp = self.dataset.index_KvsAll("valid", "sp")
        self.valid_po = self.dataset.index_KvsAll("valid", "po")

        if self.eval_data == "test":
            self.triples = self.dataset.test
        else:
            self.triples = self.dataset.valid
        if self.eval_data == "test" or self.filter_valid_with_test:
            self.test_sp = self.dataset.index_KvsAll("test", "sp")
            self.test_po = self.dataset.index_KvsAll("test", "po")

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

        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # let's go
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

            # Now update the histograms of of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filtered_valid_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test, o_ranks_filt_test, _, _ = self._filter_and_rank(
                    s, p, o, scores_sp_filt, scores_po_filt, test_labels
                )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        s_ranks_filt_test,
                        o_ranks_filt_test,
                        job=self,
                    )

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
                        event="example_rank",
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
                        event="example_rank",
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )

            # now compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filtered_valid_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # optionally: trace batch metrics
            if self.trace_batch:
                self.trace(
                    event="batch_completed",
                    type="entity_ranking",
                    scope="batch",
                    data=self.eval_data,
                    epoch=self.epoch,
                    batch=batch_number,
                    size=len(batch),
                    batches=len(self.loader),
                    **metrics,
                )

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

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filtered_valid_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filtered_valid_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
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
            event="eval_completed",
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

        for f in self.post_valid_hooks:
            f(self, trace_entry)

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

    def _get_rank(self, scores, answers):
        """Returns the rank of each answer (mean rank on ties, rounded up).

        `scores` is batch_size x entities matrix of scores. `answers` is a vector (of
        size batch_size) holding the index of the true answer in each row of `scores`.
        Scores are interpreted in descending order (rank 0 = largest score).

        If there are ties, returns the mean rank rounded up to next integer. `NaN`
        values are treated as lowest possible score (i.e., equivalent to -infinity).

        """
        # process NaN values and extract scores of true answers
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = scores[range(answers.size(0)), answers.long()]

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        num_ranks_greater = torch.sum(
            scores > true_scores.view(-1, 1), dim=1, dtype=torch.long
        )
        num_ranks_equal = torch.sum(
            scores == true_scores.view(-1, 1), dim=1, dtype=torch.long
        )

        # all done, compute (mean) ranks
        ranks = num_ranks_greater + num_ranks_equal // 2
        return ranks

    def _compute_metrics(self, rank_hist, suffix=""):
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (torch.cumsum(rank_hist[: max(self.hits_at_k_s)], dim=0) / n).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics
