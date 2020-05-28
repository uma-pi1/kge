import unittest
import os
import torch
from kge import Dataset, Config
from kge.misc import kge_base_dir


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = "dataset_test"
        self.dataset_folder = os.path.join(kge_base_dir(), "tests", "data", self.dataset_name)
        self.config = Config()
        self.config.folder = None
        self.config.set("verbose", False)
        self.config.set("model", "complex")
        self.config._import("complex")
        self.config.set("dataset.name", self.dataset_name)
        self.config.set("job.device", "cpu")
        self.splits = ["train", "valid", "test"]
        self.remove_pickle_files()

    def tearDown(self) -> None:
        self.remove_pickle_files()

    def remove_pickle_files(self):
        dataset_files = os.listdir(self.dataset_folder)
        for item in dataset_files:
            if item.endswith(".pckl"):
                os.remove(os.path.join(self.dataset_folder, item))

    def test_store_data_pickle(self):
        # this will create new pickle files for train, valid, test
        dataset = Dataset.create(config=self.config, folder=self.dataset_folder, preload_data=True)
        cache_filenames = ["train.del-t.pckl", "valid.del-t.pckl", "test.del-t.pckl"]
        for filename in cache_filenames:
            self.assertTrue(os.path.isfile(os.path.join(self.dataset_folder, filename)))

    def test_store_index_pickle(self):
        dataset = Dataset.create(config=self.config, folder=self.dataset_folder, preload_data=True)
        for split in self.splits:
            sp_o_indexname = f"{split}_sp_to_o"
            sp_o_filename = f"index-{split}_sp_to_o.pckl"
            po_s_indexname = f"{split}_po_to_s"
            po_s_filename = f"index-{split}_po_to_s.pckl"
            dataset.index(sp_o_indexname)
            dataset.index(po_s_indexname)
            self.assertTrue(
                os.path.isfile(os.path.join(self.dataset_folder, sp_o_filename))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(self.dataset_folder, po_s_filename))
            )

    def test_data_pickle_correctness(self):
        # this will create new pickle files for train, valid, test
        dataset = Dataset.create(config=self.config, folder=self.dataset_folder, preload_data=True)

        # create new dataset which loads the triples from stored pckl files
        dataset_load_by_pickle = Dataset.create(config=self.config, folder=self.dataset_folder, preload_data=True)
        for split in self.splits:
            self.assertTrue(
                torch.all(
                    torch.eq(dataset_load_by_pickle.split(split), dataset.split(split))
                )
            )

    def test_index_pickle_correctness(self):
        def _create_dataset_and_indexes():
            data = Dataset.create(config=self.config, folder=self.dataset_folder, preload_data=True)
            indexes = []
            for split in self.splits:
                sp_o_indexname = f"{split}_sp_to_o"
                po_s_indexname = f"{split}_po_to_s"
                indexes.append(data.index(sp_o_indexname))
                indexes.append(data.index(po_s_indexname))
                return data, indexes

        # this will create new pickle files for train, valid, test
        dataset, dataset_indexes = _create_dataset_and_indexes()

        # create new dataset. This will load the triples from stored pickle files
        # from previous dataset creation
        (
            dataset_load_by_pickle,
            dataset_indexes_by_pickle,
        ) = _create_dataset_and_indexes()

        for index, index_by_pickle in zip(dataset_indexes, dataset_indexes_by_pickle):
            # assert keys equal
            for key, key_by_pickle in zip(index.keys(), index_by_pickle.keys()):
                self.assertEqual(key, key_by_pickle)

            # assert values equal
            for value, value_by_pickle in zip(index.values(), index_by_pickle.values()):
                self.assertEqual(value, value_by_pickle)
