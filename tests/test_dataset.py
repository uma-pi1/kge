import unittest
import os
import torch
from tests.util import create_config, get_dataset_folder
from kge import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = "dataset_test"
        self.dataset_folder = get_dataset_folder(self.dataset_name)
        self.config = create_config(self.dataset_name)

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
        dataset = Dataset.create(
            config=self.config, folder=self.dataset_folder, preload_data=True
        )
        pickle_filenames = [
            "train.del-t.pckl",
            "valid.del-t.pckl",
            "test.del-t.pckl",
            "entity_ids.del-True-t-False.pckl",
            "relation_ids.del-True-t-False.pckl",
        ]
        for filename in pickle_filenames:
            self.assertTrue(
                os.path.isfile(os.path.join(self.dataset_folder, filename)),
                msg=filename,
            )

    def test_store_index_pickle(self):
        dataset = Dataset.create(
            config=self.config, folder=self.dataset_folder, preload_data=True
        )
        for index_key in dataset.index_functions.keys():
            dataset.index(index_key)
            pickle_filename = os.path.join(
                self.dataset_folder,
                Dataset._to_valid_filename(f"index-{index_key}.pckl"),
            )
            self.assertTrue(
                os.path.isfile(os.path.join(self.dataset_folder, pickle_filename)),
                msg=pickle_filename,
            )

    def test_data_pickle_correctness(self):
        # this will create new pickle files for train, valid, test
        dataset = Dataset.create(
            config=self.config, folder=self.dataset_folder, preload_data=True
        )

        # create new dataset which loads the triples from stored pckl files
        dataset_load_by_pickle = Dataset.create(
            config=self.config, folder=self.dataset_folder, preload_data=True
        )
        for split in dataset._triples.keys():
            self.assertTrue(
                torch.all(
                    torch.eq(dataset_load_by_pickle.split(split), dataset.split(split))
                )
            )
        self.assertEqual(dataset._meta, dataset_load_by_pickle._meta)

    def test_index_pickle_correctness(self):
        def _create_dataset_and_indexes():
            data = Dataset.create(
                config=self.config, folder=self.dataset_folder, preload_data=True
            )
            indexes = []
            for index_key in data.index_functions.keys():
                indexes.append(data.index(index_key))
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
                if type(value) is torch.Tensor:
                    self.assertTrue(torch.all(torch.eq(value, value_by_pickle)))
                else:
                    self.assertEqual(value, value_by_pickle)
