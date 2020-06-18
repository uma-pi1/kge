#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one SPO triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_ids.del" and "relation_ids.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Metadata information is stored in a file
"dataset.yaml".

"""

import argparse
import yaml
import os.path
import numpy as np

from util import process_raw_split_files
from util import process_split
from util import store_map
from util import write_obj_meta
from util import write_split_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--order_sop", action="store_true")
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")
    raw_split_files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}

    # define all split types that are derived from the raw split files
    # the key of the dicts refers to the raw split

    splits = {
        "train": {"file_key": "train", "file_name": "train.del"},
        "valid": {"file_key": "valid", "file_name": "valid.del"},
        "test": {"file_key": "test", "file_name": "test.del"},
    }

    splits_wo_unseen = {
        "valid": {
            "file_key": "valid_without_unseen",
            "file_name": "valid_without_unseen.del",
        },
        "test": {
            "file_key": "test_without_unseen",
            "file_name": "test_without_unseen.del",
        },
    }

    splits_samples = {
        "train": {"file_key": "train_sample", "file_name": "train_sample.del"}
    }

    string_files = {
        "entity_strings": "entity_strings.del",
        "relation_strings": "relation_strings.del",
    }
    split_sizes = {}

    # read data and collect entities and relations
    (
        raw_split_sizes,
        raw,
        entities,
        relations,
        entities_in_train,
        relations_in_train,
    ) = process_raw_split_files(raw_split_files, args.folder, args.order_sop)

    print(f"{len(relations)} distinct relations")
    print(f"{len(entities)} distinct entities")
    print("Writing relation and entity map...")
    store_map(relations, os.path.join(args.folder, "relation_ids.del"))
    store_map(entities, os.path.join(args.folder, "entity_ids.del"))
    print("Done.")

    # write out triples using indexes
    print("Writing triples...")

    # process train split
    process_split(
        entities,
        relations,
        file=os.path.join(args.folder, splits["train"]["file_name"]),
        raw_split=raw["train"],
        order_sop=args.order_sop,
        create_sample=True,
        sample_file=os.path.join(args.folder, splits_samples["train"]["file_name"]),
        sample_size=raw_split_sizes["valid"],
    )
    split_sizes["train"] = raw_split_sizes["train"]
    split_sizes[splits_samples["train"]["file_key"]] = raw_split_sizes["valid"]

    # process valid and test splits
    for split in ["valid", "test"]:
        size_wo_unseen = process_split(
            entities,
            relations,
            file=os.path.join(args.folder, splits[split]["file_name"]),
            raw_split=raw[split],
            order_sop=args.order_sop,
            create_filtered=True,
            filtered_file=os.path.join(
                args.folder, splits_wo_unseen[split]["file_name"]
            ),
            filter_entities=entities_in_train,
            filter_relations=relations_in_train,
        )
        split_sizes[splits_wo_unseen[split]["file_key"]] = size_wo_unseen
        split_sizes[splits[split]["file_key"]] = raw_split_sizes[split]

    # write config
    print("Writing dataset.yaml...")
    dataset_config = dict(
        name=args.folder, num_entities=len(entities), num_relations=len(relations),
    )

    # update dataset config with relation/entity maps
    write_obj_meta(dataset_config)

    # update dataset config with the meta data of the derived splits
    write_split_meta(
        [splits, splits_wo_unseen, splits_samples], dataset_config, split_sizes,
    )
    # write entity mention files
    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            dataset_config[f"files.{string}.filename"] = string_files.get(string)
            dataset_config[f"files.{string}.type"] = "idmap"

    print(yaml.dump(dict(dataset=dataset_config)))
    with open(os.path.join(args.folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))
