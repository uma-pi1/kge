import math
import time
import sys

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "entity_ranking.tie_handling.type",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("entity_ranking.tie_handling.type")

        self.tie_atol = float(self.config.get("entity_ranking.tie_handling.atol"))
        self.tie_rtol = float(self.config.get("entity_ranking.tie_handling.rtol"))

        self.filter_with_test = config.get("entity_ranking.filter_with_test")
        self.filter_splits = self.config.get("entity_ranking.filter_splits")
        if self.eval_split not in self.filter_splits:
            self.filter_splits.append(self.eval_split)

        max_k = min(
            self.dataset.num_entities(),
            max(self.config.get("entity_ranking.hits_at_k_s")),
        )
        self.hits_at_k_s = list(
            filter(lambda x: x <= max_k, self.config.get("entity_ranking.hits_at_k_s"))
        )

        #: Whether to create additional histograms for head and tail slot
        self.head_and_tail = config.get("entity_ranking.metrics_per.head_and_tail")

        #: Hooks after computing the ranks for each batch entry.
        #: Signature: hists, s, p, o, s_ranks, o_ranks, job, **kwargs
        self.hist_hooks = [hist_all]
        if config.get("entity_ranking.metrics_per.relation_type"):
            self.hist_hooks.append(hist_per_relation_type)
        if config.get("entity_ranking.metrics_per.argument_frequency"):
            self.hist_hooks.append(hist_per_frequency_percentile)

        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        """Construct all indexes needed to run."""

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        for split in self.filter_splits:
            self.dataset.index(f"{split}_sp_to_o")
            self.dataset.index(f"{split}_po_to_s")
        if "test" not in self.filter_splits and self.filter_with_test:
            self.dataset.index("test_sp_to_o")
            self.dataset.index("test_po_to_s")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        label_coords = []
        batch = torch.cat(batch).reshape((-1, 3))
        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_sp_to_o"),
                self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index("test_sp_to_o"),
                self.dataset.index("test_po_to_s"),
            )
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        return batch, label_coords, test_label_coords

    @torch.no_grad()
    def _evaluate(self):
        num_entities = self.dataset.num_entities()

        # we also filter with test data if requested
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # dictionary that maps entry of rankings to a sparse tensor containing the
        # true labels for this option
        labels_for_ranking = defaultdict(lambda: None)

        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = dict(
                type="entity_ranking",
                scope="batch",
                split=self.eval_split,
                filter_splits=self.filter_splits,
                epoch=self.epoch,
                batch=batch_number,
                size=len(batch_coords[0]),
                batches=len(self.loader),
            )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            label_coords = batch_coords[1].to(self.device)
            if filter_with_test:
                test_label_coords = batch_coords[2].to(self.device)
                # create sparse labels tensor
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),
                    2 * num_entities,
                    test_label_coords,
                    self.device,
                    float("Inf"),
                )
                labels_for_ranking["_filt_test"] = test_labels

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # compute true scores beforehand, since we can't get them from a chunked
            # score table
            # o_true_scores = self.model.score_spo(s, p, o, "o").view(-1)
            # s_true_scores = self.model.score_spo(s, p, o, "s").view(-1)
            # scoring with spo vs sp and po can lead to slight differences for ties
            # due to floating point issues.
            # We use score_sp and score_po to stay consistent with scoring used for
            # further evaluation.
            unique_o, unique_o_inverse = torch.unique(o, return_inverse=True)
            o_true_scores = torch.gather(
                self.model.score_sp(s, p, unique_o),
                1,
                unique_o_inverse.view(-1, 1),
            ).view(-1)
            unique_s, unique_s_inverse = torch.unique(s, return_inverse=True)
            s_true_scores = torch.gather(
                self.model.score_po(p, o, unique_s),
                1,
                unique_s_inverse.view(-1, 1),
            ).view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("entity_ranking.chunk_size") > -1:
                chunk_size = self.config.get("entity_ranking.chunk_size")
            else:
                chunk_size = self.dataset.num_entities()

            # process chunk by chunk
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                # compute scores of chunk
                scores = self.model.score_sp_po(
                    s, p, o, torch.arange(chunk_start, chunk_end, device=self.device)
                )
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start :]

                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()

                # check that scoring is consistent up to configured tolerance
                # if this is not the case, evaluation metrics may be artificially inflated
                close_check = torch.allclose(
                    scores_sp[o_in_chunk_mask, o_in_chunk],
                    o_true_scores[o_in_chunk_mask],
                    rtol=self.tie_rtol,
                    atol=self.tie_atol,
                )
                close_check &= torch.allclose(
                    scores_po[s_in_chunk_mask, s_in_chunk],
                    s_true_scores[s_in_chunk_mask],
                    rtol=self.tie_rtol,
                    atol=self.tie_atol,
                )
                if not close_check:
                    diff_a = torch.abs(
                        scores_sp[o_in_chunk_mask, o_in_chunk]
                        - o_true_scores[o_in_chunk_mask]
                    )
                    diff_b = torch.abs(
                        scores_po[s_in_chunk_mask, s_in_chunk]
                        - s_true_scores[s_in_chunk_mask]
                    )
                    diff_all = torch.cat((diff_a, diff_b))
                    self.config.log(
                        f"Tie-handling: mean difference between scores was: {diff_all.mean()}."
                    )
                    self.config.log(
                        f"Tie-handling: max difference between scores was: {diff_all.max()}."
                    )
                    error_message = "Error in tie-handling. The scores assigned to a triple by the SPO and SP_/_PO scoring implementations were not 'equal' given the configured tolerances. Verify the model's scoring implementations or consider increasing tie-handling tolerances."
                    if self.config.get("entity_ranking.tie_handling.warn_only"):
                        print(error_message, file=sys.stderr)
                    else:
                        raise ValueError(error_message)

                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking], chunk_start, chunk_end
                        )

                        # remove current example from labels
                        labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
                        labels_chunk[
                            s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
                        ] = 0

                    # compute partial ranking and filter the scores (sets scores of true
                    # labels to infinity)
                    (
                        s_rank_chunk,
                        s_num_ties_chunk,
                        o_rank_chunk,
                        o_num_ties_chunk,
                        scores_sp_filt,
                        scores_po_filt,
                    ) = self._filter_and_rank(
                        scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
                    )

                    # from now on, use filtered scores
                    scores_sp = scores_sp_filt
                    scores_po = scores_po_filt

                    # update rankings
                    ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
                    ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
                    ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
                    ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

                # we are done with the chunk

            # We are done with all chunks; calculate final ranks from counts
            s_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["s_raw"][0],
                ranks_and_ties_for_ranking["s_raw"][1],
            )
            o_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["o_raw"][0],
                ranks_and_ties_for_ranking["o_raw"][1],
            )
            s_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["s_filt"][0],
                ranks_and_ties_for_ranking["s_filt"][1],
            )
            o_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["o_filt"][0],
                ranks_and_ties_for_ranking["o_filt"][1],
            )

            # Update the histograms of of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filter_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["s_filt_test"][0],
                    ranks_and_ties_for_ranking["s_filt_test"][1],
                )
                o_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["o_filt_test"][0],
                    ranks_and_ties_for_ranking["o_filt_test"][1],
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
                    "split": self.eval_split,
                    "filter_splits": self.filter_splits,
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
                    if filter_with_test:
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
                    if filter_with_test:
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

            # Compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # update batch trace with the results
            self.current_trace["batch"].update(metrics)

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # output batch information to console
            self.config.print(
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
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
        epoch_time += time.time()

        # update trace with results
        self.current_trace["epoch"].update(
            dict(epoch_time=epoch_time, event="eval_completed", **metrics,)
        )

    def _densify_chunk_of_labels(
        self, labels: torch.Tensor, chunk_start: int, chunk_end: int
    ) -> torch.Tensor:
        """Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the lower
        bound and 'chunk_end' the upper bound.

        The resulting tensor contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the batch
        for sp and po

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
        the po chunk.

        """
        num_entities = self.dataset.num_entities()
        indices = labels._indices()
        mask_sp = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
        mask_po = ((chunk_start + num_entities) <= indices[1, :]) & (
            indices[1, :] < (chunk_end + num_entities)
        )
        indices_sp_chunk = indices[:, mask_sp]
        indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - chunk_start
        indices_po_chunk = indices[:, mask_po]
        indices_po_chunk[1, :] = (
            indices_po_chunk[1, :] - num_entities - chunk_start * 2 + chunk_end
        )
        indices_chunk = torch.cat((indices_sp_chunk, indices_po_chunk), dim=1)
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            # since all sparse label tensors have the same value we could also
            # create a new tensor here without indexing with:
            # torch.full([indices_chunk.shape[1]], float("inf"), device=self.device)
            labels._values()[mask_sp | mask_po],
            torch.Size([labels.size()[0], (chunk_end - chunk_start) * 2]),
        ).to_dense()
        return dense_labels

    def _filter_and_rank(
        self,
        scores_sp: torch.Tensor,
        scores_po: torch.Tensor,
        labels: torch.Tensor,
        o_true_scores: torch.Tensor,
        s_true_scores: torch.Tensor,
    ):
        """Filters the current examples with the given labels and returns counts rank and
num_ties for each true score.

        :param scores_sp: batch_size x chunk_size tensor of scores

        :param scores_po: batch_size x chunk_size tensor of scores

        :param labels: batch_size x 2*chunk_size tensor of scores

        :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
        objects in batch

        :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
        subjects in batch

        :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
        scores_sp and scores_po

        """
        chunk_size = scores_sp.shape[1]
        if labels is not None:
            # remove current example from labels
            labels_sp = labels[:, :chunk_size]
            labels_po = labels[:, chunk_size:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_rank, o_num_ties = self._get_ranks_and_num_ties(scores_sp, o_true_scores)
        s_rank, s_num_ties = self._get_ranks_and_num_ties(scores_po, s_true_scores)
        return s_rank, s_num_ties, o_rank, o_num_ties, scores_sp, scores_po

    def _get_ranks_and_num_ties(
        self, scores: torch.Tensor, true_scores: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """
        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        is_close = torch.isclose(
            scores, true_scores.view(-1, 1), rtol=self.tie_rtol, atol=self.tie_atol
        )
        is_greater = scores > true_scores.view(-1, 1)
        num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
        rank = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than the one of
        the true score

        :param num_ties: batch_size x tensor with number of scores equal as the one of
        the true score

        :return: batch_size x 1 tensor of ranks

        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """Computes desired matrix from rank histogram"""
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities() + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (
                torch.cumsum(
                    rank_hist[: max(self.hits_at_k_s)], dim=0, dtype=torch.float64
                )
                / n
            ).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics


# HISTOGRAM COMPUTATION ###############################################################


def __initialize_hist(hists, key, job):
    """If there is no histogram with given `key` in `hists`, add an empty one."""
    if key not in hists:
        hists[key] = torch.zeros(
            [job.dataset.num_entities()],
            device=job.config.get("job.device"),
            dtype=torch.float,
        )


def hist_all(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    """Create histogram of all subject/object ranks (key: "all").

    `hists` a dictionary of histograms to update; only key "all" will be affected. `s`,
    `p`, `o` are true triples indexes for the batch. `s_ranks` and `o_ranks` are the
    rank of the true answer for (?,p,o) and (s,p,?) obtained from a model.

    """
    __initialize_hist(hists, "all", job)
    if job.head_and_tail:
        __initialize_hist(hists, "head", job)
        __initialize_hist(hists, "tail", job)
        hist_head = hists["head"]
        hist_tail = hists["tail"]

    hist = hists["all"]
    o_ranks_unique, o_ranks_count = torch.unique(o_ranks, return_counts=True)
    s_ranks_unique, s_ranks_count = torch.unique(s_ranks, return_counts=True)
    hist.index_add_(0, o_ranks_unique, o_ranks_count.float())
    hist.index_add_(0, s_ranks_unique, s_ranks_count.float())
    if job.head_and_tail:
        hist_tail.index_add_(0, o_ranks_unique, o_ranks_count.float())
        hist_head.index_add_(0, s_ranks_unique, s_ranks_count.float())


def hist_per_relation_type(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    for rel_type, rels in job.dataset.index("relations_per_type").items():
        __initialize_hist(hists, rel_type, job)
        hist = hists[rel_type]
        if job.head_and_tail:
            __initialize_hist(hists, f"{rel_type}_head", job)
            __initialize_hist(hists, f"{rel_type}_tail", job)
            hist_head = hists[f"{rel_type}_head"]
            hist_tail = hists[f"{rel_type}_tail"]

        mask = [_p in rels for _p in p.tolist()]
        for r, m in zip(o_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_tail[r] += 1

        for r, m in zip(s_ranks, mask):
            if m:
                hists[rel_type][r] += 1
                if job.head_and_tail:
                    hist_head[r] += 1


def hist_per_frequency_percentile(hists, s, p, o, s_ranks, o_ranks, job, **kwargs):
    # initialize
    frequency_percs = job.dataset.index("frequency_percentiles")
    for arg, percs in frequency_percs.items():
        for perc, value in percs.items():
            __initialize_hist(hists, "{}_{}".format(arg, perc), job)

    # go
    for perc in frequency_percs["subject"].keys():  # same for relation and object
        for r, m_s, m_r in zip(
            s_ranks,
            [id in frequency_percs["subject"][perc] for id in s.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_s:
                hists["{}_{}".format("subject", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
        for r, m_o, m_r in zip(
            o_ranks,
            [id in frequency_percs["object"][perc] for id in o.tolist()],
            [id in frequency_percs["relation"][perc] for id in p.tolist()],
        ):
            if m_o:
                hists["{}_{}".format("object", perc)][r] += 1
            if m_r:
                hists["{}_{}".format("relation", perc)][r] += 1
