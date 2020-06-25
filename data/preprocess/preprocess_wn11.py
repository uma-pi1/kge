#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict

from util import analyze_raw_splits
from util import process_split
from util import process_pos_neg_split
from util import write_split_meta
from util import RawDataset
from util import write_dataset_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")
    raw_split_files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
    raw_split_sizes = {}

    # define all split types that are derived from the raw split files
    # the key of the dicts refers to the raw split

    splits_positives = {
        "train": {"file_key": "train", "file_name": "train.del"},
        "valid": {"file_key": "valid", "file_name": "valid.del"},
        "test": {"file_key": "test", "file_name": "test.del"},
    }

    splits_positives_wo_unseen = {
        "valid": {
            "file_key": "valid_without_unseen",
            "file_name": "valid_without_unseen.del",
        },
        "test": {
            "file_key": "test_without_unseen",
            "file_name": "test_without_unseen.del",
        },
    }

    splits_negatives = {
        "valid": {"file_key": "valid_negatives", "file_name": "valid_negatives.del"},
        "test": {"file_key": "test_negatives", "file_name": "test_negatives.del"},
    }

    splits_negatives_wo_unseen = {
        "valid": {
            "file_key": "valid_negatives_without_unseen",
            "file_name": "valid_negatives_without_unseen.del",
        },
        "test": {
            "file_key": "test_negatives_without_unseen",
            "file_name": "test_negatives_without_unseen.del",
        },
    }

    splits_samples = {
        "train": {"file_key": "train_sample", "file_name": "train_sample.del"}
    }

    # read data and collect entities and relations; additionally processes metadata
    dataset: RawDataset = analyze_raw_splits(
        raw_split_files=raw_split_files,
        folder=args.folder,
        collect_objects_in=["train"],
    )

    # update dataset config with derived splits
    write_split_meta(
        [
            splits_positives,
            splits_positives_wo_unseen,
            splits_negatives,
            splits_negatives_wo_unseen,
            splits_samples,
        ],
        dataset.config,
    )

    # process the training splits and write triples
    process_split(
        "train",
        dataset,
        file_name=splits_positives["train"]["file_name"],
        file_key=splits_positives["train"]["file_key"],
        create_sample=True,
        sample_size=dataset.raw_split_sizes["valid"],
        sample_file=splits_samples["train"]["file_name"],
        sample_key=splits_samples["train"]["file_key"],
    )

    # process the valid/test splits and write triples
    for split in ["valid", "test"]:
        process_pos_neg_split(
            split,
            dataset,
            pos_file=splits_positives[split]["file_name"],
            pos_key=splits_positives[split]["file_key"],
            neg_file=splits_negatives[split]["file_name"],
            neg_key=splits_negatives[split]["file_key"],
            create_filtered=True,
            filtered_pos_file=splits_positives_wo_unseen[split]["file_name"],
            filtered_pos_key=splits_positives_wo_unseen[split]["file_key"],
            filtered_neg_file=splits_negatives_wo_unseen[split]["file_name"],
            filtered_neg_key=splits_negatives_wo_unseen[split]["file_key"],
            filtered_include_ent=dataset.entities_in_split["train"],
            filtered_include_rel=dataset.relations_in_split["train"],
        )

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
