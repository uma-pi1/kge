#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict

from util import store_map
from util import analyze_raw_splits
from util import write_triple
from util import process_split
from util import process_pos_neg_split
from util import write_obj_meta
from util import write_split_meta

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

    # read data and collect entities and relations
    (
        raw_split_sizes,
        raw,
        entities,
        relations,
        entities_in_train,
        relations_in_train,
    ) = analyze_raw_splits(raw_split_files, args.folder)

    split_sizes = {}

    print(f"{len(relations)} distinct relations")
    print(f"{len(entities)} distinct entities")
    print("Writing relation and entity map...")
    store_map(relations, os.path.join(args.folder, "relation_ids.del"))
    store_map(entities, os.path.join(args.folder, "entity_ids.del"))
    print("Done.")

    # write out triples using indexes
    print("Writing triples...")

    # process the training split
    train_file = os.path.join(args.folder, splits_positives["train"]["file_name"])

    process_split(
        entities,
        relations,
        file=train_file,
        raw_split=raw["train"],
        create_sample=True,
        sample_file=splits_samples["train"]["file_name"],
        sample_size=raw_split_sizes["valid"],
    )

    split_sizes["train"] = raw_split_sizes["train"]
    split_sizes[splits_samples["train"]["file_key"]] = raw_split_sizes["valid"]
    # process the valid and test splits
    for split in ["valid", "test"]:

        (
            pos_size,
            filtered_pos_size,
            neg_size,
            filtered_neg_size,
        ) = process_pos_neg_split(
            entities,
            relations,
            pos_file=os.path.join(args.folder, splits_positives[split]["file_name"]),
            neg_file=os.path.join(args.folder, splits_negatives[split]["file_name"]),
            raw_split=raw[split],
            create_filtered=True,
            filtered_pos_file=os.path.join(
                args.folder, splits_positives_wo_unseen[split]["file_name"]
            ),
            filtered_neg_file=os.path.join(
                args.folder, splits_negatives_wo_unseen[split]["file_name"]
            ),
            filter_entities=entities_in_train,
            filter_relations=relations_in_train,
        )

        split_sizes[splits_positives[split]["file_key"]] = pos_size
        split_sizes[splits_negatives[split]["file_key"]] = neg_size
        split_sizes[splits_positives_wo_unseen[split]["file_key"]] = filtered_pos_size
        split_sizes[splits_negatives_wo_unseen[split]["file_key"]] = filtered_neg_size

    # write config
    print("Writing dataset.yaml...")
    dataset_config = dict(
        name=args.folder, num_entities=len(entities), num_relations=len(relations),
    )

    # update dataset config with relation/entity maps
    write_obj_meta(dataset_config)

    # update dataset config with the meta data of the splits
    write_split_meta(
        [
            splits_positives,
            splits_positives_wo_unseen,
            splits_negatives,
            splits_negatives_wo_unseen,
            splits_samples,
        ],
        dataset_config,
        split_sizes,
    )

    print(yaml.dump(dict(dataset=dataset_config)))
    with open(os.path.join(args.folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))
