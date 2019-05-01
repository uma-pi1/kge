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
        self.triples = self.dataset.valid
        if self.eval_data == "test" or self.filter_valid_with_test:
            self.triples = self.dataset.test
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

        self.is_prepared = True

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        train_label_coords = kge.job.util.get_batch_sp_po_coords(
            batch, self.dataset.num_entities, self.train_sp, self.train_po
        )
        valid_label_coords = kge.job.util.get_batch_sp_po_coords(
            batch, self.dataset.num_entities, self.valid_sp, self.valid_po
        )
        if self.eval_data == "test" or self.filter_valid_with_test:
            test_label_coords = kge.job.util.get_batch_sp_po_coords(
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

        # these histograms will gather information about how the ranks of the
        # correct answer (raw rank and filtered rank)
        hist = torch.zeros([num_entities], device=self.device, dtype=torch.float)
        hist_filt = torch.zeros([num_entities], device=self.device, dtype=torch.float)

        # we also filter with test data during validation if requested
        evil = self.eval_data == "valid" and self.filter_valid_with_test
        if evil:
            hist_filt_test = torch.zeros(
                [num_entities], device=self.device, dtype=torch.float
            )

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
                if evil:
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

            # compute raw ranks rank and histogram of raw ranks
            batch_hist, s_ranks, o_ranks, _, _ = self._filter_and_rank(
                s, p, o, scores_sp, scores_po, None
            )

            # same for filtered ranks
            batch_hist_filt, s_ranks_filt, o_ranks_filt, scores_sp_filt, scores_po_filt = self._filter_and_rank(
                s, p, o, scores_sp, scores_po, labels
            )

            # and for filtered_with_test ranks
            if evil:
                batch_hist_filt_test, s_ranks_filt_test, o_ranks_filt_test, _, _ = self._filter_and_rank(
                    s, p, o, scores_sp_filt, scores_po_filt, test_labels
                )

            # update global rank histograms
            hist += batch_hist
            hist_filt += batch_hist_filt
            if evil:
                hist_filt_test += batch_hist_filt_test

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
                    if evil:
                        entry["rank_filtered_with_test"] = (
                            o_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry
                    )
                    if evil:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry
                    )

            # now compute the metrics
            metrics = self._compute_metrics(batch_hist)
            metrics.update(self._compute_metrics(batch_hist_filt, suffix="_filtered"))
            if evil:
                metrics.update(
                    self._compute_metrics(
                        batch_hist_filt_test, suffix="_filtered_with_test"
                    )
                )

            # optionally: trace batch metrics
            if self.trace_batch:
                self.trace(
                    type="entity_ranking",
                    scope="batch",
                    data=self.eval_data,
                    epoch=self.epoch,
                    batch=i,
                    size=len(batch),
                    batches=len(self.loader),
                    **metrics
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
        if evil:
            metrics.update(
                self._compute_metrics(hist_filt_test, suffix="_filtered_with_test")
            )
        epoch_time += time.time()

        # and trace them
        trace_entry = self.trace(
            echo=True,
            echo_prefix="  ",
            log=True,
            type="entity_ranking",
            scope="epoch",
            data=self.eval_data,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
            epoch_time=epoch_time,
            **metrics
        )

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_data + " data.")

        return trace_entry

    def _filter_and_rank(self, s, p, o, scores_sp, scores_po, labels):
        num_entities = self.dataset.num_entities
        if labels is not None:
            for i in range(len(o)):  # remove current example from labels
                labels[i, o[i]] = 0
                labels[i, num_entities + s[i]] = 0
            labels_sp = labels[:, :num_entities]
            labels_po = labels[:, num_entities:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_ranks = self._get_rank(scores_sp, o)
        s_ranks = self._get_rank(scores_po, s)
        batch_hist = torch.zeros([num_entities], device=self.device, dtype=torch.float)
        # need for loop because batch_hist[o_ranks]+=1 ignores repeated
        # entries in o_ranks
        for r in o_ranks:
            batch_hist[r] += 1
        for r in s_ranks:
            batch_hist[r] += 1
        return batch_hist, s_ranks, o_ranks, scores_sp, scores_po

    def _get_rank(self, scores, answers):
        answers = answers.reshape((-1, 1)).expand(-1, self.dataset.num_entities).long()
        true_scores = torch.gather(scores, 1, answers)
        scores = scores + 1e-40
        ranks = torch.sum((scores > true_scores).long(), dim=1)
        ranks = ranks - (ranks == self.dataset.num_entities).long()
        return ranks

    def _compute_metrics(self, rank_hist, suffix=""):
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = (
            torch.tensor(range(self.dataset.num_entities), device=self.device).float()
            + 1.0
        )
        metrics["mean_rank" + suffix] = torch.sum(rank_hist * ranks).item() / n

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            torch.sum(rank_hist * reciprocal_ranks).item() / n
        )

        metrics["hits_at_k" + suffix] = (
            torch.cumsum(rank_hist[: self.max_k], dim=0) * 1 / n
        ).tolist()

        return metrics
