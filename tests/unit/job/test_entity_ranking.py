import unittest
import torch
from kge.model import KgeModel
from kge import Dataset, Config
from kge.job import EntityRankingJob


class TestEntityRanking(unittest.TestCase):

    def setUp(self) -> None:
        self.config = Config()
        self.config.set("model", "complex")
        self.config._import("complex")
        self.config.set("dataset.name", "toy")
        self.config.set("job.device", "cpu")
        self.config.folder = "."
        self.dataset = Dataset.create(self.config)
        self.model = KgeModel.create(self.config, self.dataset)
        self.entity_ranking_job = EntityRankingJob(self.config, self.dataset, model=self.model, parent_job=None)

    def test__collate(self):
        batch = [torch.LongTensor([1,2,3]), torch.LongTensor([2,2,4]), torch.LongTensor([3,2,5])]
        collate_return = self.entity_ranking_job._collate(batch)
        self.assertTrue(torch.all(torch.eq(torch.stack(batch), collate_return[0])))
