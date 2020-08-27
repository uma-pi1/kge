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

from util import analyze_splits
from util import RawDataset
from util import DerivedSplitBase
from util import DerivedSplitSample
from util import DerivedSplitFiltered
from util import Split
from util import write_dataset_config
from util import process_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--S", action="store", default=0)
    parser.add_argument("--P", action="store", default=1)
    parser.add_argument("--O", action="store", default=2)
    args = parser.parse_args()

    S, P, O = int(args.S), int(args.P), int(args.O)

    print(f"Preprocessing {args.folder}...")

    # register all base splits
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

    # read data and collect entity and relation maps
    dataset: RawDataset = analyze_splits(
        splits=[train, valid, test],
        folder=args.folder,
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


    string_files = {
        "entity_strings": "entity_strings.del",
        "relation_strings": "relation_strings.del",
    }


    # write all splits and collect meta data
    process_splits(dataset)

    # update config with entity string files
    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            dataset.config[f"files.{string}.filename"] = string_files.get(string)
            dataset.config[f"files.{string}.type"] = "idmap"

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
