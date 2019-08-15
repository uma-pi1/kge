import torch
import time
import kge.job
from kge.job import EvaluationJob


class EntityPairRankingJob(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = False

    def _prepare(self):
        """Create dataloader and construct all indexes needed to run."""

        if self.is_prepared:
            return

        # Get set of relations in data
        if self.eval_data == "test":
            self.triples = self.dataset.test
        else:
            self.triples = self.dataset.valid
        self.relations = torch.unique(self.triples[:, 1])

        # Create data loader
        self.loader = torch.utils.data.DataLoader(
            self.relations,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

        # Create indexes and evaluation data
        # These indexes hold a list of (s,o) tuples for every relation p
        self.train_p = self.dataset.index_MtoN("train")
        self.valid_p = self.dataset.index_MtoN("valid")
        self.test_p = self.dataset.index_MtoN("test")
        if self.eval_data == "test":
            self.triples = self.dataset.test
        else:
            self.triples = self.dataset.valid

        # Let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):
        """
        For filtering: gets positions of positive triples in train (and valid)
        for all relations in the batch, stores them in filtering_coords.
        For computing metrics: gets positions of positive triples in test
        for all relations in the batch, stores them in test_triples_coords.
        Positions of test triples are flattened for later convenience.
        Returns batch, filtering_coords and test_triples_coords.
        """

        filtering_coords = kge.job.util.get_batch_so_coords(batch, self.train_p)

        if self.eval_data == "test":
            valid_coords = kge.job.util.get_batch_so_coords(batch, self.valid_p)
            filtering_coords = torch.cat(filtering_coords, valid_coords)

        test_triples_coords = kge.job.util.get_batch_so_coords(batch, self.test_p)

        # TODO flatten test_triples_coords

        return torch.tensor(batch), filtering_coords, test_triples_coords

    # TODO:
    # 1. Get 3-way tensor of scores from scores_p function of the model
    #   For filtering, make known triples' scores -Inf here.
    #   For this you need positions of known triples for each relation (train set, or train + valid).
    #   These positions should come from indexes which take a relation and return a set of (s,o) tuples
    #   These indexes should be used by the collate function to build the filtering_coords.
    #   Then use filtering_coords from collate to build a filt tensor of zeros the same size as the scores tensor.
    #       See how ER does it wrt being sparse and stuff, i.e. their tensor called labels.
    #   Make all entries of known positives in this filt tensor be Inf.
    #       Loop over relations in batch, then set their collection of (s,o) tuples to -Inf
    #       See similar multi indexing logic for filtering in current implementation of ER
    #   Then do scores_filt = scores - filt
    #   Every point that follows will have to be done for both scores and scores_filt tensors
    # 2. Turn the 3-way scores tensor into matrix of (batch_size, num_entities^2)
    #   This should be scores.view(batch_size, -1), I think, check by printing scores.size after this.
    #   This is needed because torch.top_k can only get top entries in a single dimension
    #   But we need two, because we are after top entries in the score matrix
    # 3. Use torch.topk(matrix, k=self.max_k, dim=1) to get top k entries per relation in batch
    # 4. Store output of torch.topk in top_k
    # 5. Then, top_k.indices is (batch_size, self.max_k) and contains indices of top k per relation (row)
    #   These indices range is [0, num_entities^2], because they are indices of a flattened score matrix
    # 6. Then match these indices with test triple indices for given relation
    #   Ideally, the test triple indices would also be from a flattened score matrix.
    #   If not, you'd need to unravel (turn to 2D) the ones from topk, or flatten the ones from the test triples
    #   These could come from an index called test_triples.
    #   It could take relations as key and return a list of (flattened or 2d tuples) positions of its test triples
    #   These would be the positions of the test triples in the score matrix of a given relation.
    #   I think in the creation of this index we need to map tne (s,o) tuples (i.e. 2d indexes) to 1D.
    #   Meaning, do the work here to convert these to 1d already, so that matching here is easy.
    # 7. Count matches for hits, do what is needed for MAP.
    #   The entries in topk.indices are already the indices of the sorted top k entries.
    #   Meaning the first index is of the max scored triple, the second index of the second max, and so on.
    #   This should be useful for computing the MAP@k.

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on " + self.eval_data + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities

        # These histograms will gather information about the ranks of the
        # correct answer (raw rank and filtered rank)
        # TODO for PR I think these should be size k because we only sort the top k
        # Confirm when implementing
        hist = torch.zeros([self.max_k], device=self.device, dtype=torch.float)
        hist_filt = torch.zeros([self.max_k], device=self.device, dtype=torch.float)

        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # construct a label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            batch_size = batch.size(0)
            filtering_coords = batch_coords[1].to(self.device)
            test_triples_coords = batch_coords[2].to(self.device)
            labels = kge.job.util.coord_to_sparse_tensor(
                num_entities,
                num_entities,
                filtering_coords,
                self.device,
                float("Inf"),
                depth=batch_size
            ).to_dense()

            # compute all scores
            scores = self.model.score_p(batch[0])

            # compute raw ranks rank and histogram of raw ranks
            batch_hist, topk, _ = self._get_topk(scores, batch_size, None)

            # same for filtered ranks
            batch_hist_filt, topk_filt, scores_filt = self._get_topk(scores, batch_size, labels)

            # update global rank histograms
            hist += batch_hist
            hist_filt += batch_hist_filt

            # optionally: trace ranks of each example
            if self.trace_examples:
                entry = {
                    "type": "entity_pair_ranking",
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
                    self.trace(
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry,
                    )
                    if evil:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )

            # now compute the metrics
            metrics = self._compute_metrics(batch_hist)
            metrics.update(self._compute_metrics(batch_hist_filt, suffix="_filtered"))

            # optionally: trace batch metrics
            if self.trace_batch:
                self.trace(
                    type="entity_pair_ranking",
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
                    metrics["hits_at_k"][0],
                    metrics["hits_at_k_filtered"][0],
                    self.max_k,
                    metrics["hits_at_k"][self.max_k - 1],
                    metrics["hits_at_k_filtered"][self.max_k - 1],
                    ),
                end="",
                flush=True,
            )

        # we are done; compute final metrics
        print("\033[2K\r", end="", flush=True)  # clear line and go back
        metrics = self._compute_metrics(hist)
        metrics.update(self._compute_metrics(hist_filt, suffix="_filtered"))
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="entity_pair_ranking",
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
                {"config": self.config, **trace_entry},
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_data + " data.")

        return trace_entry

    # 2. Turn the 3-way scores tensor into matrix of (batch_size, num_entities^2)
    #   This should be scores.view(batch_size, -1), I think, check by printing scores.size after this.
    #   This is needed because torch.top_k can only get top entries in a single dimension
    #   But we need two, because we are after top entries in the score matrix
    # 3. Use torch.topk(matrix, k=self.max_k, dim=1) to get top k entries per relation in batch
    # 4. Store output of torch.topk in top_k
    # 5. Then, top_k.indices is (batch_size, self.max_k) and contains indices of top k per relation (row)
    #   These indices range is [0, num_entities^2], because they are indices of a flattened score matrix

    def _get_topk(self, scores, batch_size, labels):
        num_entities = self.dataset.num_entities

        # TODO FILTERING!
        # Understand this and adapt for PR
        # if labels is not None:
        #     for i in range(len(o)):  # remove current example from labels
        #         labels[i, o[i]] = 0
        #         labels[i, num_entities + s[i]] = 0
        #     scores = scores - labels

        # Turn scores tensor to 2d
        # This is needed because torch.top_k can only get top entries in a single dimension
        # But we need two, because we are after top entries in the score matrix
        scores = scores.view(batch_size, -1)
        topk = torch.topk(scores, k=self.max_k, dim=1)

        # Compute batch histogram
        o_ranks = self._get_rank(scores_sp, o)
        batch_hist = torch.zeros([self.max_k], device=self.device, dtype=torch.float)
        # Need for loop because batch_hist[o_ranks]+=1 ignores repeated entries
        for r in o_ranks:
            batch_hist[r] += 1

        return batch_hist, topk, scores
