#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess_base.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one field_map triple per line, separated by tabs.

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

import util

if __name__ == "__main__":
    args = util.default_parser().parse_args()
    field_map = {
        "S": args.subject_field,
        "P": args.predicate_field,
        "O": args.object_field,
    }

    print(f"Preprocessing {args.folder}...")

    # register raw splits
    train_raw = util.RawSplit(
        file="train.txt",
        field_map=field_map,
        collect_entities=True,
        collect_relations=True,
    )
    valid_raw = util.RawSplit(file="valid.txt", field_map=field_map,)
    test_raw = util.RawSplit(file="test.txt", field_map=field_map,)

    # create raw dataset with default splits
    raw_dataset = util.create_raw_dataset(train_raw, valid_raw, test_raw, args)

    # do the work
    util.process_splits(raw_dataset)
    util.update_string_files(raw_dataset, args)
    util.write_dataset_yaml(raw_dataset.config, args.folder)
