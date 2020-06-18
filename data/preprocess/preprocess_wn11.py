#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict

from util import store_map
from util import process_raw_split_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")
    raw_split_files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
    raw_split_sizes = {}

    # create mappings from the keys of raw_split_files to derived split types

    splits_positives = {
        "train": {"file_key": "train", "file_name": "train.del"},
        "valid": {"file_key": "valid", "file_name": "valid.del"},
        "test": {"file_key": "test", "file_name": "test.del"},
    }

    splits_positives_wo_unseen = {
        "train": {"file_key": "train_sample", "file_name": "train_sample.del"},
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
    # read data and collect entities and relations
    (
        raw_split_sizes,
        raw,
        entities,
        relations,
        entities_in_train,
        relations_in_train,
    ) = process_raw_split_files(raw_split_files, args.folder)

    split_sizes = {}
    S, P, O = 0, 1, 2

    print(f"{len(relations)} distinct relations")
    print(f"{len(entities)} distinct entities")
    print("Writing relation and entity map...")
    store_map(relations, os.path.join(args.folder, "relation_ids.del"))
    store_map(entities, os.path.join(args.folder, "entity_ids.del"))
    print("Done.")

    # write out triples using indexes
    print("Writing triples...")
    for split in ["train", "valid", "test"]:
        with open(
            os.path.join(args.folder, splits_positives[split]["file_name"]), "w"
        ) as f_pos, open(
            os.path.join(args.folder, splits_positives_wo_unseen[split]["file_name"]),
            "w",
        ) as f_pos_wo_unseen:
            # for valid and test additionally create split files with negative examples
            if split in ["valid", "test"]:
                f_neg = open(
                    os.path.join(args.folder, splits_negatives[split]["file_name"]),
                    "w",
                )
                f_neg_wo_unseen = open(
                    os.path.join(
                        args.folder, splits_negatives_wo_unseen[split]["file_name"]
                    ),
                    "w",
                )
            else:
                # for train use f_pos_wo_unseen to create a sampled train split
                train_sample = np.random.choice(
                    raw_split_sizes[split], raw_split_sizes["valid"], False
                )
            # create sizes of the derived splits
            size_negatives = 0
            size_negatives_wo_unseen = 0
            size_positives = 0
            size_positives_wo_unseen = 0
            for n, t in enumerate(raw[split]):
                # write the ordinary split files
                if split in ["valid", "test"] and int(t[3]) == -1:
                    file_wrapper = f_neg
                    size_negatives += 1
                else:
                    size_positives += 1
                    file_wrapper = f_pos
                file_wrapper.write(
                    str(entities[t[S]])
                    + "\t"
                    + str(relations[t[P]])
                    + "\t"
                    + str(entities[t[O]])
                    + "\n"
                )
                # write files w/o unseen and train sample
                if split == "train" and n in train_sample:
                    f_pos_wo_unseen.write(
                        str(entities[t[S]])
                        + "\t"
                        + str(relations[t[P]])
                        + "\t"
                        + str(entities[t[O]])
                        + "\n"
                    )
                    size_positives_wo_unseen += 1
                elif (
                    split in ["valid", "test"]
                    and t[S] in entities_in_train
                    and t[O] in entities_in_train
                    and t[P] in relations_in_train
                ):

                    if int(t[3]) == -1:
                        file_wrapper = f_neg_wo_unseen
                        size_negatives_wo_unseen += 1
                    else:
                        file_wrapper = f_pos_wo_unseen
                        size_positives_wo_unseen += 1

                    file_wrapper.write(
                        str(entities[t[S]])
                        + "\t"
                        + str(relations[t[P]])
                        + "\t"
                        + str(entities[t[O]])
                        + "\n"
                    )

                split_sizes[splits_positives[split]["file_key"]] = size_positives
                split_sizes[
                    splits_positives_wo_unseen[split]["file_key"]
                ] = size_positives_wo_unseen
                if split in ["valid", "test"]:
                    split_sizes[
                        splits_negatives_wo_unseen[split]["file_key"]
                    ] = size_negatives_wo_unseen
                    split_sizes[splits_negatives[split]["file_key"]] = size_negatives

    # write config
    print("Writing dataset.yaml...")
    dataset_config = dict(
        name=args.folder, num_entities=len(entities), num_relations=len(relations),
    )
    for obj in ["entity", "relation"]:
        dataset_config[f"files.{obj}_ids.filename"] = f"{obj}_ids.del"
        dataset_config[f"files.{obj}_ids.type"] = "map"

    for split_type in [
            splits_positives,
            splits_positives_wo_unseen,
            splits_negatives,
            splits_negatives_wo_unseen,
        ]:
        for raw_split, split_dict in split_type.items():
            file_key = split_dict["file_key"]
            file_name = split_dict["file_name"]
            dataset_config[f"files.{file_key}.filename"] = file_name
            dataset_config[f"files.{file_key}.type"] = "triples"
            dataset_config[f"files.{file_key}.size"] = split_sizes[file_key]

    print(yaml.dump(dict(dataset=dataset_config)))
    with open(os.path.join(args.folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))
