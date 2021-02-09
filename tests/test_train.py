import unittest
import torch
import math
from tests.util import create_config, get_dataset_folder
from kge import Dataset
from kge.model import KgeModel
from kge.indexing import KvsAllIndex
from kge.job.train import TrainingJob
import tempfile
import random
import numpy as np, numba
import numpy.random

# tests for all models
class BaseTestTrain:
    def __init__(self, train_type, options={}):
        self.train_type = train_type
        self.dataset_name = "dataset_test"
        self.options = options

    def setUp(self):
        self.config = create_config(self.dataset_name)
        self.config.set_all({"lookup_embedder.dim": 32})
        self.config.set("job.type", "train")
        self.config.set("train.type", self.train_type)
        self.config.set_all(self.options)
        self.dataset_folder = get_dataset_folder(self.dataset_name)
        self.dataset = Dataset.create(
            self.config, folder=get_dataset_folder(self.dataset_name)
        )
        self.model = KgeModel.create(self.config, self.dataset)

    def test_subbatches(self):
        avg_losses = torch.empty(2)
        for i, subbatch_size in enumerate([-1, 3]):
            from kge.util.seed import seed_all
            seed_all(0)

            with tempfile.TemporaryDirectory() as tmpdirname:
                config = self.config.clone()
                config.folder = tmpdirname
                config.set("train.subbatch_size", subbatch_size)
                job = TrainingJob.create(
                    config=config,
                    dataset=self.dataset,
                    model=self.model,
                    forward_only=True,
                )
                job._prepare()
                trace = job.run_epoch()
                avg_losses[i] = trace["avg_loss"]

        self.assertTrue(torch.isclose(avg_losses[0], avg_losses[1]))

# instances for each training job type
class Test1vsAll(BaseTestTrain, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestTrain.__init__(self, "1vsAll")

class TestKvsAll(BaseTestTrain, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestTrain.__init__(self, "KvsAll")


class TestNegativeSampling(BaseTestTrain, unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName=methodName)
        BaseTestTrain.__init__(self, "negative_sampling")
