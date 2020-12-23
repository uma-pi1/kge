import torch
import kge.job
from kge.job.entity_ranking import EntityRankingJob
from kge import Config, Dataset
from kge.job import EvaluationJob, Job

class OLPEntityRankingJob(EntityRankingJob):

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == OLPEntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # load the olp quintuples
        self.triples = self.dataset.split_olp(self.config.get("eval.split"))
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

        # handle alternative subjects and alternative objects
        if len(batch) == 3:
            alternative_subjects = list(batch[1])
            alternative_objects = list(batch[2])
            batch = list(batch[0])

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

        batch = torch.cat(batch).reshape((-1, 3))
        return batch, label_coords, test_label_coords
