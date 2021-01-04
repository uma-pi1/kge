import math
import time
from operator import itemgetter

import torch
import kge.job
from kge.job.entity_ranking import EntityRankingJob
from kge import Config, Dataset
from kge.job import EvaluationJob, Job
from collections import defaultdict


class OLPEntityRankingJob(EntityRankingJob):

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == OLPEntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        self.loader = torch.utils.data.DataLoader(
            range(self.triples.shape[0]),
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        split = self.config.get("eval.split")

        label_coords = []

        batch_data = torch.index_select(self.triples, 0, torch.tensor(batch))

        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch_data,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_sp_to_o"),
                self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch_data,
                self.dataset.num_entities(),
                self.dataset.index("test_sp_to_o"),
                self.dataset.index("test_po_to_s"),
            )
        else:
            test_label_coords = torch.zeros([0, 2], dtype=torch.long)

        # batch_data = torch.cat(batch_data).reshape((-1, 3))

        alternative_subject_mentions = torch.cat(
            itemgetter(*batch)(self.dataset._alternative_subject_mentions[split]))
        alternative_object_mentions = torch.cat(itemgetter(*batch)(self.dataset._alternative_object_mentions[split]))

        return batch_data, label_coords, test_label_coords, alternative_subject_mentions, alternative_object_mentions

    def compute_true_scores(self, batch_coords):
        alternative_subject_mentions = batch_coords[3].to(self.device)
        alternative_object_mentions = batch_coords[4].to(self.device)
        o_true_scores_all_mentions = self.model.score_spo(alternative_object_mentions[:, 0],
                                                          alternative_object_mentions[:, 1],
                                                          alternative_object_mentions[:, 3], "o").view(-1)
        s_true_scores_all_mentions = self.model.score_spo(alternative_subject_mentions[:, 3],
                                                          alternative_subject_mentions[:, 1],
                                                          alternative_subject_mentions[:, 2], "s").view(-1)

        # inspired by https://github.com/pytorch/pytorch/issues/36748#issuecomment-620279304
        def filter_mention_results(scores, quadruples):
            ranks = torch.unique_consecutive(quadruples[:, 0:3], dim=0, return_inverse=True)[1]
            score_sort_index = scores.sort(descending=True)[1]
            ranks_by_score = ranks[score_sort_index]
            unique_ranks, unique_ranks_inverse = torch.unique(ranks_by_score, sorted=True, return_inverse=True,
                                                              dim=0)
            perm = torch.arange(unique_ranks_inverse.size(0), dtype=unique_ranks_inverse.dtype,
                                device=unique_ranks_inverse.device)
            unique_ranks_inverse, perm = unique_ranks_inverse.flip([0]), perm.flip([0])
            rank_index = unique_ranks_inverse.new_empty(unique_ranks.size(0)).scatter_(0, unique_ranks_inverse,
                                                                                       perm)

            return scores[score_sort_index[rank_index]]

        o_true_scores = filter_mention_results(o_true_scores_all_mentions, alternative_object_mentions)
        s_true_scores = filter_mention_results(s_true_scores_all_mentions, alternative_subject_mentions)
        return o_true_scores, s_true_scores
