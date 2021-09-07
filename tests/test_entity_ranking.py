import unittest
import tempfile
import yaml

import torch
from kge.job import EntityRankingJob
from torch import Tensor

from kge import Dataset
from kge.model import KgeModel

from tests.util import create_config, get_dataset_folder


class TestModel(KgeModel):
    """
    Used to test proper tie handling, returns the same score for all triples.
    Optionally returns slightly different scores depending on which scoring function is used.
    """

    SCORE = 7.291993    # arbitrary score used for all triples
    SPO_INC = 1e-6      # arbitrary small increment for SPO triples

    def __init__(self, config, dataset, spo_different):
        """
        Args:
            spo_different: Whether or not SPO scores are incremented with SPO_INC. This is done to simulate a difference
             in calculation method (e.g. for TransE SPO is calculated with F.pairwise_distance, but for SP_ / _PO
             torch.cdist is used, which results in small differences).
        """
        super().__init__(config=config, dataset=dataset, scorer=None, create_embedders=False)

        self.spo_different = spo_different

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        score = self.SCORE
        if self.spo_different:
            score += self.SPO_INC
        score = torch.tensor([score]).expand_as(s)
        return score

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None) -> Tensor:
        score = torch.tensor([self.SCORE])
        score = score.expand(len(s), len(o))
        return score

    def score_po(self, p: Tensor, o: Tensor, s: Tensor = None) -> Tensor:
        score = torch.tensor([self.SCORE])
        score = score.expand(len(o), len(s))
        return score

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, entity_subset: Tensor = None) -> Tensor:
        num_entities = self.dataset.num_entities()
        if entity_subset is None:
            entity_subset = torch.arange(num_entities)

        a = self.score_sp(s, p, entity_subset)
        b = self.score_po(p, o, entity_subset)
        return torch.cat((a, b), dim=1)

    def prepare_job(self, job: "Job", **kwargs):
        pass


class BaseTestEntityRanking:

    def __init__(self, spo_different, options={}):
        self.spo_different = spo_different
        self.dataset_name = "dataset_test"
        self.options = options

    def setUp(self):
        config = create_config(self.dataset_name)
        config.set_all(self.options)
        config.set_all({"eval.trace_level": "example"})
        config.folder = "."
        config.log_folder = tempfile.mkdtemp(prefix="kge-")

        dataset = Dataset.create(
            config, folder=get_dataset_folder(self.dataset_name)
        )

        model = TestModel(config, dataset, self.spo_different)
        self.job = EntityRankingJob(config=config, dataset=dataset, parent_job=None, model=model)

    def test_tie_handling(self):

        # run the evaluation
        self.job.run()

        # since all entities get the same score, num_ties should equal num_entities, and rank should be 0
        num_ties = self.job.dataset.num_entities()
        rank = 0

        # read trace file and check ranks are what they should be
        trace_file = self.job.config.tracefile()
        with open(trace_file, 'r') as file:
            yaml_lines = file.readlines()

        for yaml_line in yaml_lines:
            yaml_line = yaml.safe_load(yaml_line)
            if yaml_line['job_id'] == self.job.job_id \
                    and 'event' in yaml_line.keys() and yaml_line['event'] == "example_rank":
                self.assertEqual(self.job._get_ranks(rank, num_ties) + 1, yaml_line['rank'])


class TestEntityRankingSPODifferent(BaseTestEntityRanking, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestEntityRanking.__init__(self, spo_different=True)


class TestEntityRankingSPOSame(BaseTestEntityRanking, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestEntityRanking.__init__(self, spo_different=False)
