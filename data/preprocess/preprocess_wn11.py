#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict

from util import analyze_raw_splits
from util import RawDataset
from util import write_dataset_config
from util import RawSplit
from util import PosNegRawSplit
from util import process_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")

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

    raw_valid = PosNegRawSplit(
        file="valid.txt",
        derived_split_pos_key="valid",
        derived_split_pos_options={
            "type": "triples",
            "filename": "valid.del",
            "split_type": "valid",
        },
        derived_split_neg_key="valid_negative",
        derived_split_neg_options={
            "type": "triples",
            "filename": "valid_negative.del",
            "split_type": "valid",
        },
        derived_split_filtered_pos_key="valid_without_unseen",
        filtered_split_pos_options={
            "type": "triples",
            "filename": "valid_without_unseen.del",
            "split_type": "valid",
        },
        derived_split_filtered_neg_key="valid_without_unseen_negative",
        filtered_split_neg_options={
            "type": "triples",
            "filename": "valid_without_unseen_negative.del",
            "split_type": "valid",
        },
        filter_with=raw_train,
    )

    raw_test = PosNegRawSplit(
        file="test.txt",
        derived_split_pos_key="test",
        derived_split_pos_options={
            "type": "triples",
            "filename": "test.del",
            "split_type": "test",
        },
        derived_split_neg_key="test_negative",
        derived_split_neg_options={
            "type": "triples",
            "filename": "test_negative.del",
            "split_type": "test",
        },
        derived_split_filtered_pos_key="test_without_unseen",
        filtered_split_pos_options={
            "type": "triples",
            "filename": "test_without_unseen.del",
            "split_type": "test",
        },
        derived_split_filtered_neg_key="test_without_unseen_negative",
        filtered_split_neg_options={
            "type": "triples",
            "filename": "test_without_unseen_negative.del",
            "split_type": "test",
        },
        filter_with=raw_train,
    )

    # read data and collect entities and relations
    dataset: RawDataset = analyze_raw_splits(
        raw_splits=[raw_train, raw_valid, raw_test],
        folder=args.folder,
    )
    raw_train.sample_size = raw_valid.size

    # write all splits and collect meta data
    process_splits(dataset)

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
