import unittest
import os
import torch
from tests.util import create_config, empty_cache, get_cache_dir
from kge.misc import kge_base_dir
from kge.model.kge_model import KgeModel
from kge.job import TrainingJob
from kge.dataset import Dataset


class TestFreeze(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = "toy"
        self.folder = os.path.join(get_cache_dir(), "test_freeze")
        self.config = create_config(self.dataset_name)
        self.config.folder = self.folder
        self.config.init_folder()
        self.config.set("train.max_epochs", 1)
        self.dataset = Dataset.create(config=self.config)

    def tearDown(self) -> None:
        empty_cache()

    def test_freeze(self) -> None:
        """Test if frozen embeddings are correctly frozen.

           Ensure, after calling freeze() of the LookupEmbedder, embeddings are hold
            constant during training.

        """

        model = KgeModel.create(config=self.config, dataset=self.dataset)

        # freeze every other entity and relation embedding
        freeze_indexes_ent = list(range(0, model.dataset.num_entities(), 2))
        freeze_indexes_rel = list(range(0, model.dataset.num_relations(), 2))

        entity_embedder = model.get_o_embedder()
        relation_embedder = model.get_p_embedder()

        # copy before freeze
        frozen_emb_rel = (
            relation_embedder.embed(torch.tensor(freeze_indexes_rel)).clone().detach()
        )

        frozen_emb_ent = (
            entity_embedder.embed(torch.tensor(freeze_indexes_ent)).clone().detach()
        )

        # freeze
        entity_embedder.freeze(freeze_indexes_ent)
        relation_embedder.freeze(freeze_indexes_rel)

        training_job = TrainingJob.create(
            config=model.config, dataset=model.dataset, model=model
        )
        training_job.run()

        frozen_emb_rel_after = relation_embedder.embed(torch.tensor(freeze_indexes_rel))
        frozen_emb_ent_after = entity_embedder.embed(torch.tensor(freeze_indexes_ent))

        # Ensure the frozen embeddings have not been changed
        self.assertTrue(
            torch.all(torch.eq(frozen_emb_ent, frozen_emb_ent_after)),
            msg="Frozen parameter changed during training",
        )

        self.assertTrue(
            torch.all(torch.eq(frozen_emb_rel, frozen_emb_rel_after)),
            msg="Frozen parameter changed during training",
        )

    def test_scores_after_freeze(self) -> None:
        """Test if score calculation is correct after calling freeze() on Embeddings."""

        model = KgeModel.create(config=self.config, dataset=self.dataset)

        # freeze every other entity and relation embedding
        freeze_indexes_ent = list(range(0, model.dataset.num_entities(), 2))
        freeze_indexes_rel = list(range(0, model.dataset.num_relations(), 2))

        entity_embedder = model.get_o_embedder()
        relation_embedder = model.get_p_embedder()

        triples = self.dataset.split("train")
        scores_before = model.score_spo(triples[:, 0], triples[:, 1], triples[:, 2])

        entity_embedder.freeze(freeze_indexes_ent)
        relation_embedder.freeze(freeze_indexes_rel)

        scores_after = model.score_spo(triples[:, 0], triples[:, 1], triples[:, 2])

        self.assertTrue(
            torch.all(torch.eq(scores_before, scores_after)),
            msg="Model score computation has changed after calling freeze."
        )
