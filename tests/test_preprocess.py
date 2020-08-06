import unittest
from tests.util import get_dataset_folder
import sys
from kge.misc import kge_base_dir
import os
from os import path

sys.path.append(path.join(kge_base_dir(), "data/preprocess"))
from data.preprocess.util import analyze_raw_splits
from data.preprocess.util import RawSplit
from data.preprocess.util import RawDataset
from data.preprocess.util import process_splits
from data.preprocess.util import write_dataset_config
import yaml


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = "dataset_preprocess"
        self.dataset_folder = get_dataset_folder(self.dataset_name)

    def tearDown(self) -> None:
        self.remove_del_files()

    def test_analyze_raw_splits(self):
        raw_splits = TestPreprocess.get_raw_splits()
        dataset: RawDataset = analyze_raw_splits(
            raw_splits=list(raw_splits.values()), folder=self.dataset_folder,
        )

        # check if objects are collected correctly
        self.assertTrue(
            all(
                [
                    rel in dataset.all_relations.keys()
                    for rel in ["r1", "r2", "r3", "r4"]
                ]
            )
        )
        self.assertTrue(
            all([ent in dataset.all_entities.keys() for ent in ["a", "b", "c", "d"]])
        )

        # check entity/relation index for uniqueness
        entity_index = list(dataset.all_entities.values())
        self.assertEqual(entity_index, list(set(entity_index)))
        relation_index = list(dataset.all_relations.values())
        self.assertEqual(relation_index, list(set(relation_index)))

        # check entity/relation index for completeness and erroneous entries
        for index in [entity_index, relation_index]:
            length = len(index)
            correct_index = list(range(length))
            self.assertEqual(index, correct_index)

        # check if entity/relation maps have been written
        self.assertTrue(
            os.path.isfile(os.path.join(self.dataset_folder, "entity_ids.del"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.dataset_folder, "relation_ids.del"))
        )

        # check sizes of the raw data
        self.assertTrue(raw_splits["raw_train"].size == 6)
        self.assertTrue(raw_splits["raw_valid"].size == 5)
        self.assertTrue(raw_splits["raw_test"].size == 4)

    def test_write_splits(self):
        raw_splits = TestPreprocess.get_raw_splits()
        dataset: RawDataset = analyze_raw_splits(
            raw_splits=list(raw_splits.values()), folder=self.dataset_folder,
        )
        raw_splits["raw_train"].sample_size = 3

        # check all files have been created and sizes are tracked correctly
        for split in dataset.raw_splits:
            self._test_write_splits(split, dataset)

        # explicitly check if filtering is correct
        raw_test = raw_splits["raw_test"]
        filtered_options = raw_test.filtered_split_options
        filename = filtered_options["filename"]
        f_path = os.path.join(self.dataset_folder, filename)
        with open(f_path, "r") as f:
            triples = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            for triple in triples:
                # the index of the unseen relation and entity is 3 respectively (d, r4)
                self.assertFalse(triple[0] == 3)
                self.assertFalse(triple[1] == 3)
                self.assertFalse(triple[2] == 3)

    def _test_write_splits(self, split, dataset):
        split.write_splits(dataset.all_entities, dataset.all_relations, dataset.folder)
        for key, options in [
            (split.derived_split_key, split.derived_split_options),
            (split.derived_sample_split_key, split.sample_split_options),
            (split.derived_filtered_split_key, split.filtered_split_options),
        ]:
            if key:
                filename = options["filename"]
                f_path = os.path.join(self.dataset_folder, filename)
                # check correct file has been written
                self.assertTrue(os.path.isfile(f_path))
                with open(f_path, "r") as f:
                    # check the correct size has been tracked
                    data = f.readlines()
                    self.assertTrue(options["size"] == len(data))

    def test_write_dataset_config(self):
        # check if the dataset.yaml file has been written as expected
        raw_splits = TestPreprocess.get_raw_splits()
        dataset: RawDataset = analyze_raw_splits(
            raw_splits=list(raw_splits.values()), folder=self.dataset_folder,
        )
        raw_splits["raw_train"].sample_size = 3
        process_splits(dataset)
        write_dataset_config(dataset.config, self.dataset_folder)

        yaml_path = os.path.join(self.dataset_folder, "dataset.yaml")
        self.assertTrue(os.path.isfile(yaml_path))

        with open(yaml_path, "r") as yaml_file:
            options = yaml.load(yaml_file, Loader=yaml.SafeLoader)["dataset"]

            # check expected values for significant keys

            self.assertTrue(options["files.train.size"] == 6)
            self.assertTrue(options["files.valid.size"] == 5)
            self.assertTrue(options["files.test.size"] == 4)

            self.assertTrue(options["files.valid_without_unseen.size"] == 2)
            self.assertTrue(options["files.test_without_unseen.size"] == 1)
            self.assertTrue(options["files.train_sample.size"] == 3)

            self.assertTrue(options["num_entities"] == 4)
            self.assertTrue(options["num_relations"] == 4)

        os.remove(yaml_path)

    def remove_del_files(self):
        files = os.listdir(self.dataset_folder)
        for item in files:
            if item.endswith(".del"):
                os.remove(os.path.join(self.dataset_folder, item))

    @staticmethod
    def get_raw_splits():
        raw_train = RawSplit(
            file="train.txt",
            collect_entities=True,
            collect_relations=True,
            derived_split_key="train",
            derived_split_options={
                "type": "triples",
                "filename": "train.del",
                "split_type": "train",
            },
            derived_sample_split_key="train_sample",
            sample_split_options={
                "type": "triples",
                "filename": "train_sample.del",
                "split_type": "train",
            },
        )
        raw_valid = RawSplit(
            file="valid.txt",
            derived_split_key="valid",
            derived_split_options={
                "type": "triples",
                "filename": "valid.del",
                "split_type": "valid",
            },
            derived_filtered_split_key="valid_without_unseen",
            filter_with=raw_train,
            filtered_split_options={
                "type": "triples",
                "filename": "valid_without_unseen.del",
                "split_type": "valid",
            },
        )
        raw_test = RawSplit(
            file="test.txt",
            derived_split_key="test",
            derived_split_options={
                "type": "triples",
                "filename": "test.del",
                "split_type": "test",
            },
            derived_filtered_split_key="test_without_unseen",
            filter_with=raw_train,
            filtered_split_options={
                "type": "triples",
                "filename": "test_without_unseen.del",
                "split_type": "test",
            },
        )
        return {"raw_train": raw_train, "raw_valid": raw_valid, "raw_test": raw_test}
