import math
import time
from collections import defaultdict

import torch
import kge.job

from kge.job import EntityRankingJob


class EntityRankingJobCpuGpuSwitcher(EntityRankingJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model, embedding_cpu_switcher):
        super().__init__(config, dataset, parent_job, model)
        self.embedding_cpu_switcher = embedding_cpu_switcher

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

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"]
            if filtered_valid_with_test
            else ["_raw", "_filt"]
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

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            train_label_coords = batch_coords[1].to(self.device)
            valid_label_coords = batch_coords[2].to(self.device)
            test_label_coords = batch_coords[3].to(self.device)
            if self.eval_data == "test":
                label_coords = torch.cat(
                    [train_label_coords, valid_label_coords, test_label_coords]
                )
            elif self.eval_data == "valid":  # it's valid
                label_coords = torch.cat([train_label_coords, valid_label_coords])
                if filtered_valid_with_test:
                    # create sparse labels tensor
                    test_labels = kge.job.util.coord_to_sparse_tensor(
                        len(batch),
                        2 * num_entities,
                        test_label_coords,
                        self.device,
                        float("Inf"),
                    )
                    labels_for_ranking["_filt_test"] = test_labels
            else:  # neither test nor valid
                raise ValueError()

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # move embeddings of batch to gpu
            s = s.clone()
            o = o.clone()
            self.embedding_cpu_switcher.create_indexes(torch.unique(torch.cat((s, o), dim=0)), self.device)
            self.embedding_cpu_switcher.to_gpu()
            s_mapped = self.embedding_cpu_switcher.map_indexes(s, self.device)
            o_mapped = self.embedding_cpu_switcher.map_indexes(o, self.device)
            s = s.clone()
            o = o.clone()

            # compute true scores beforehand, since we can't get them from a chunked
            # score table
            o_true_scores = self.model.score_spo(s_mapped, p, o_mapped, "o").view(-1)
            s_true_scores = self.model.score_spo(s_mapped, p, o_mapped, "s").view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("eval.chunk_size") > -1:
                chunk_size = self.config.get("eval.chunk_size")
            else:
                chunk_size = self.dataset.num_entities

            # process chunk by chung
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                # make sure that embeddings of the chunk and batch are on the gpu
                unique_entities = torch.unique(torch.cat((s.long(), o.long(),
                                                          torch.arange(chunk_start, chunk_end).to(self.device).long()),
                                                         dim=0))
                self.embedding_cpu_switcher.create_indexes(unique_entities, self.device)
                self.embedding_cpu_switcher.to_gpu()
                s_mapped = self.embedding_cpu_switcher.map_indexes(s, self.device)
                o_mapped = self.embedding_cpu_switcher.map_indexes(o, self.device)
                chunk_index_mapped = self.embedding_cpu_switcher.map_indexes(torch.arange(chunk_start, chunk_end),
                                                                             self.device)

                # compute scores of chunk
                scores = self.model.score_sp_po(
                    s_mapped, p, o_mapped, chunk_index_mapped
                )
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start:]

                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()
                scores_sp[o_in_chunk_mask, o_in_chunk] = o_true_scores[o_in_chunk_mask]
                scores_po[s_in_chunk_mask, s_in_chunk] = s_true_scores[s_in_chunk_mask]

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
                    s_rank_chunk, s_num_ties_chunk, o_rank_chunk, o_num_ties_chunk, scores_sp_filt, scores_po_filt = self._filter_and_rank(
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
            if filtered_valid_with_test:
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

            # Compute the batch metrics for the full histogram (key "all")
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

    def resume(self, checkpoint_file=None):
        """Load model state from last or specified checkpoint."""
        # load model
        from kge.job import TrainingJob

        training_job = TrainingJob.create(self.config, self.dataset)
        training_job.resume(checkpoint_file)
        self.model = training_job.model
        self.embedding_cpu_switcher = training_job.embedding_cpu_switcher
        self.epoch = training_job.epoch
        self.resumed_from_job_id = training_job.resumed_from_job_id
        self.trace(
            event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint_file
        )
