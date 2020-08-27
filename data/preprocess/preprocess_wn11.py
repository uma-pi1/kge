#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict

from util import analyze_splits
from util import RawDataset
from util import write_dataset_config
from util import Split
from util import DerivedSplitBase
from util import DerivedSplitSample
from util import DerivedLabeledSplit
from util import DerivedLabeledSplitFiltered
from util import process_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--S", action="store", default=0)
    parser.add_argument("--P", action="store", default=1)
    parser.add_argument("--O", action="store", default=2)
    args = parser.parse_args()

    SPO = {"S": int(args.S), "P": int(args.P), "O": int(args.O)}

    # register all base splits
    train = Split(
        file="train.txt", SPO=SPO, collect_entities=True, collect_relations=True,
    )

    valid = Split(file="valid.txt", SPO=SPO,)

    test = Split(file="test.txt", SPO=SPO,)

    # read data and collect entity and relation maps
    dataset: RawDataset = analyze_splits(
        splits=[train, valid, test], folder=args.folder,
    )

    # register all splits to be derived from the base splits
    # arbitrary options can be added which will be written the the dataset config

    train_derived = DerivedSplitBase(
        parent_split=train,
        key="train",
        options={"type": "triples", "filename": "train.del", "split_type": "train"},
    )

    train_derived_sample = DerivedSplitSample(
        parent_split=train,
        key="train_sample",
        sample_size=len(valid.raw_data),
        options={
            "type": "triples",
            "filename": "train_sample.del",
            "split_type": "train",
        },
    )

    train.derived_splits.extend([train_derived, train_derived_sample])

    valid_pos_derived = DerivedLabeledSplit(
        parent_split=valid,
        key="valid",
        options={"type": "triples", "filename": "valid.del", "split_type": "valid"},
        label=1,
    )

    valid_neg_derived = DerivedLabeledSplit(
        parent_split=valid,
        key="valid_negatives",
        options={
            "type": "triples",
            "filename": "valid_negatives.del",
            "split_type": "valid",
        },
        label=-1,
    )

    valid_pos_wo_unseen_derived = DerivedLabeledSplitFiltered(
        parent_split=valid,
        key="valid_without_unseen",
        filter_with=train,
        options={
            "type": "triples",
            "filename": "valid_without_unseen.del",
            "split_type": "valid",
        },
        label=1,
    )

    valid_neg_wo_unseen_derived = DerivedLabeledSplitFiltered(
        parent_split=valid,
        key="valid_without_unseen_negatives",
        filter_with=train,
        options={
            "type": "triples",
            "filename": "valid_without_unseen_negatives.del",
            "split_type": "valid",
        },
        label=-1,
    )

    valid.derived_splits.extend(
        [
            valid_pos_derived,
            valid_neg_derived,
            valid_pos_wo_unseen_derived,
            valid_neg_wo_unseen_derived,
        ]
    )

    test_pos_derived = DerivedLabeledSplit(
        parent_split=test,
        key="test",
        options={"type": "triples", "filename": "test.del", "split_type": "test"},
        label=1,
    )

    test_neg_derived = DerivedLabeledSplit(
        parent_split=test,
        key="test_negatives",
        options={
            "type": "triples",
            "filename": "test_negatives.del",
            "split_type": "test",
        },
        label=-1,
    )

    test_pos_wo_unseen_derived = DerivedLabeledSplitFiltered(
        parent_split=test,
        key="test_without_unseen",
        filter_with=train,
        options={
            "type": "triples",
            "filename": "test_without_unseen.del",
            "split_type": "test",
        },
        label=1,
    )

    test_neg_wo_unseen_derived = DerivedLabeledSplitFiltered(
        parent_split=test,
        key="test_without_unseen_negatives",
        filter_with=train,
        options={
            "type": "triples",
            "filename": "test_without_unseen_negatives.del",
            "split_type": "test",
        },
        label=-1,
    )

    test.derived_splits.extend(
        [
            test_pos_derived,
            test_neg_derived,
            test_pos_wo_unseen_derived,
            test_neg_wo_unseen_derived,
        ]
    )

    # write all splits and collect meta data
    process_splits(dataset)

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
