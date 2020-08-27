import unittest
from tests.util import get_dataset_folder
import sys
from kge.misc import kge_base_dir
import os
from os import path

sys.path.append(path.join(kge_base_dir(), "data/preprocess"))
from data.preprocess.util import analyze_splits
from data.preprocess.util import RawDataset
from data.preprocess.util import DerivedSplitBase
from data.preprocess.util import DerivedSplitSample
from data.preprocess.util import DerivedSplitFiltered
from data.preprocess.util import Split
from data.preprocess.util import write_dataset_config
from data.preprocess.util import process_splits
import yaml




class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = "dataset_preprocess"
        self.dataset_folder = get_dataset_folder(self.dataset_name)

    def tearDown(self) -> None:
        self.remove_del_files()

    def test_analyze_splits(self):
        splits = TestPreprocess.get_splits()
        dataset: RawDataset = analyze_splits(
            splits=list(splits.values()), folder=self.dataset_folder,
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
        self.assertTrue(splits["train"].size == 6)
        self.assertTrue(splits["valid"].size == 5)
        self.assertTrue(splits["test"].size == 4)

    def test_write_splits(self):
        splits = TestPreprocess.get_splits()
        dataset: RawDataset = analyze_splits(
            splits=list(splits.values()), folder=self.dataset_folder,
        )
        self.set_derived_splits(splits["train"], splits["valid"], splits["test"])

        # write and check all files have been created and sizes are tracked correctly
        for split in dataset.splits:
            self._test_write_splits(split, dataset)

        # explicitly check if filtering is correct
        test = splits["test"]
        for derived_split in test.derived_splits:
            if isinstance(derived_split, DerivedSplitFiltered):
                options = derived_split.options
                filename = options["filename"]
                f_path = os.path.join(self.dataset_folder, filename)
        with open(f_path, "r") as f:
            triples = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            for triple in triples:
                # the index of the unseen relation and entity is 3 respectively (d, r4)
                # ensure this has been filtered out correctly
                self.assertFalse(triple[0] == 3)
                self.assertFalse(triple[1] == 3)
                self.assertFalse(triple[2] == 3)

    def _test_write_splits(self, split, dataset):
        split.write_splits(dataset.all_entities, dataset.all_relations, dataset.folder)
        for derived_split in split.derived_splits:
            filename = derived_split.options["filename"]
            f_path = os.path.join(self.dataset_folder, filename)
            # check correct file has been written
            self.assertTrue(os.path.isfile(f_path))
            with open(f_path, "r") as f:
                # check the correct size has been tracked
                data = f.readlines()
                self.assertTrue(derived_split.options["size"] == len(data))

    def test_write_dataset_config(self):
        # check if the dataset.yaml file has been written as expected
        splits = TestPreprocess.get_splits()
        dataset: RawDataset = analyze_splits(
            splits=list(splits.values()), folder=self.dataset_folder,
        )
        self.set_derived_splits(splits["train"], splits["valid"], splits["test"])
        process_splits(dataset)
        # write config
        write_dataset_config(dataset.config, self.dataset_folder)
        # check file has been written
        yaml_path = os.path.join(self.dataset_folder, "dataset.yaml")
        self.assertTrue(os.path.isfile(yaml_path))

        # check correctness of significant keys
        with open(yaml_path, "r") as yaml_file:
            options = yaml.load(yaml_file, Loader=yaml.SafeLoader)["dataset"]
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
    def get_splits():
        S, P, O = 0, 1, 2

        train = Split(
            file="train.txt",
            SPO={"S": S, "P": P, "O": O},
            collect_entities=True,
            collect_relations=True,
        )
        valid = Split(
            file="valid.txt",
            SPO={"S": S, "P": P, "O": O},
        )
        test = Split(
            file="test.txt",
            SPO={"S": S, "P": P, "O": O},
        )

        return {"train": train, "valid": valid, "test": test}

    def set_derived_splits(self, train: Split, valid: Split, test: Split):
        train_derived = DerivedSplitBase(
            parent_split=train,
            key="train",
            options={"type": "triples", "filename": "train.del", "split_type": "train"},
        )

        train_derived_sample = DerivedSplitSample(
            parent_split=train,
            key="train_sample",
            sample_size=3,
            options={
                "type": "triples",
                "filename": "train_sample.del",
                "split_type": "train"
            },
        )

        train.derived_splits.extend([train_derived, train_derived_sample])

        valid_derived = DerivedSplitBase(
            parent_split=valid,
            key="valid",
            options={"type": "triples", "filename": "valid.del", "split_type": "valid"},
        )

        valid_derived_wo_unseen = DerivedSplitFiltered(
            parent_split=valid,
            key="valid_without_unseen",
            filter_with=train,
            options={
                "type": "triples",
                "filename": "valid_without_unseen.del",
                "split_type": "valid"
            },
        )

        valid.derived_splits.extend([valid_derived, valid_derived_wo_unseen])

        test_derived = DerivedSplitBase(
            parent_split=test,
            key="test",
            options={"type": "triples", "filename": "test.del", "split_type": "test"},
        )

        test_derived_wo_unseen = DerivedSplitFiltered(
            parent_split=test,
            key="test_without_unseen",
            filter_with=train,
            options={
                "type": "triples",
                "filename": "test_without_unseen.del",
                "split_type": "test"
            },
        )

        test.derived_splits.extend([test_derived, test_derived_wo_unseen])
