import unittest
import torch
import math
from tests.util import create_config, get_dataset_folder
from kge import Dataset
from kge.model import KgeModel
from kge.indexing import KvsAllIndex


# tests for all models


class BaseTestModel:
    def __init__(self, model_name, options={}):
        self.model_name = model_name
        self.dataset_name = "dataset_test"
        self.options = options

    def setUp(self):
        self.config = create_config(self.dataset_name, model=self.model_name)
        self.config.set_all({"lookup_embedder.dim": 32})
        self.config.set_all(self.options)
        self.dataset_folder = get_dataset_folder(self.dataset_name)
        self.dataset = Dataset.create(
            self.config, folder=get_dataset_folder(self.dataset_name)
        )
        self.model = KgeModel.create(self.config, self.dataset)

    def test_score_equality(self):
        self.model.eval()
        num_entities = self.dataset.num_entities()
        num_relations = self.dataset.num_relations()

        # score spo results (all predictions)
        s = torch.arange(num_entities).repeat_interleave(num_relations * num_entities)
        p = (
            torch.arange(num_relations)
            .repeat_interleave(num_entities)
            .repeat(num_entities)
        )
        o = torch.arange(num_entities).repeat(num_relations * num_entities)
        score_spo_s = self.model.score_spo(s, p, o, direction="s")
        score_spo_o = self.model.score_spo(s, p, o, direction="o")

        # score sp_ (all predictions)
        s = torch.arange(num_entities).repeat_interleave(num_relations)
        p = torch.arange(num_relations).repeat(num_entities)
        # print(self.model)
        # self.model.score_spo(s, p, s, direction="o")
        score_sp = self.model.score_sp(s, p)
        self.assertTrue(
            torch.allclose(
                score_spo_o.view(-1), score_sp.view(-1), atol=1e-5, rtol=1e-4
            ),
            msg="consistency of score_spo with score_sp:\n{} \n{}".format(
                score_spo_o.view(-1), score_sp.view(-1)
            ),
        )

        # score _po (all predictions)
        p = torch.arange(num_relations).repeat_interleave(num_entities)
        o = torch.arange(num_entities).repeat(num_relations)
        score_po = self.model.score_po(p, o).t().contiguous()
        self.assertTrue(
            torch.allclose(
                score_spo_s.view(-1), score_po.view(-1), atol=1e-5, rtol=1e-4
            ),
            msg="consistency of score_spo with score_po:\n{} \n{}".format(
                score_spo_s.view(-1), score_po.view(-1)
            ),
        )


# instances for each model + model-specific tests


class TestComplEx(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "complex")


class TestConvE(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(
            self,
            "reciprocal_relations_model",
            options={"reciprocal_relations_model.base_model.type": "conve",},
        )


class TestCP(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "cp")


class TestDistMult(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "distmult")


class TestRelationalTucker3(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "relational_tucker3")


class TestRescal(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "rescal")


class TestRotatE(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "rotate")

    def test_normalize_phases(self):
        model = KgeModel.create(self.config, self.dataset)
        model.eval()
        num_entities = self.dataset.num_entities()
        num_relations = self.dataset.num_relations()

        # start with embeddings outside of [-pi,pi]
        data = model.get_p_embedder()._embeddings.weight.data
        data[:] = (torch.rand(data.shape) - 0.5) * 100

        # perform initial predictions
        s = torch.arange(num_entities).repeat_interleave(num_relations * num_entities)
        p = (
            torch.arange(num_relations)
            .repeat_interleave(num_entities)
            .repeat(num_entities)
        )
        o = torch.arange(num_entities).repeat(num_relations * num_entities)
        scores_org = model.score_spo(s, p, o)

        # now normalize phases
        model.normalize_phases()

        # check if predictions are unaffected
        scores_new = model.score_spo(s, p, o)
        self.assertTrue(
            torch.allclose(scores_org, scores_new),
            msg="test that normalizing phases does not change predictions",
        )

        # check that phases are normalized
        data = model.get_p_embedder()._embeddings.weight.data
        self.assertTrue(
            torch.all((data >= -math.pi) & (data < math.pi)),
            msg="check that phases are normalized",
        )


class TestSimplE(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "simple")


class TestTransE(BaseTestModel, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestModel.__init__(self, "transe")
