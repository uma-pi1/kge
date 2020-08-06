#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one SPO triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_ids.del" and "relation_ids.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Additionally, the splits
"train_sample.del" (a random subset of train) and "valid_without_unseen.del" and
"test_without_unseen.del" are stored. The "test/valid_without_unseen.del" files are
subsets of "valid.del" and "test.del" resp. where all triples containing entities
or relations not existing in "train.del" have been filtered out.

Metadata information is stored in a file "dataset.yaml".

"""

import argparse
import yaml
import os.path
import numpy as np

from util import analyze_raw_splits
from util import store_map
from util import RawDataset
from util import RawSplit
from util import write_dataset_config
from util import process_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--order_sop", action="store_true")
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")

    raw_train = RawSplit(
        file="train.txt",
        order_sop=args.order_sop,
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
        order_sop=args.order_sop,
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
        order_sop=args.order_sop,
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

    string_files = {
        "entity_strings": "entity_strings.del",
        "relation_strings": "relation_strings.del",
    }

    # read data and collect entity and relation maps
    dataset: RawDataset = analyze_raw_splits(
        raw_splits=[raw_train, raw_valid, raw_test],
        order_sop=args.order_sop,
        folder=args.folder,
    )
    raw_train.sample_size = raw_valid.size

    # write all splits and collect meta data
    process_splits(dataset)

    # update config with entity string files
    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            dataset.config[f"files.{string}.filename"] = string_files.get(string)
            dataset.config[f"files.{string}.type"] = "idmap"

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
