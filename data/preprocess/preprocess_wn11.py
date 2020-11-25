#!/usr/bin/env python
"""Preprocess the WN11 dataset into a the format expected by libKGE. """

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

    # create raw dataset
    raw_dataset = util.create_raw_dataset(
        train_raw, valid_raw, test_raw, args, create_splits=False
    )

    # create splits: TRAIN
    train = util.Split(
        raw_split=train_raw,
        key="train",
        options={"type": "triples", "filename": "train.del", "split_type": "train"},
    )
    train_sample = util.SampledSplit(
        raw_split=train_raw,
        key="train_sample",
        sample_size=len(valid_raw.data),
        options={
            "type": "triples",
            "filename": "train_sample.del",
            "split_type": "train",
        },
    )
    train_raw.splits.extend([train, train_sample])

    # create splits: VALID
    valid_pos = util.LabeledSplit(
        raw_split=valid_raw,
        key="valid",
        options={"type": "triples", "filename": "valid.del", "split_type": "valid"},
        label=1,
    )
    valid_neg = util.LabeledSplit(
        raw_split=valid_raw,
        key="valid_negatives",
        options={
            "type": "triples",
            "filename": "valid_negatives.del",
            "split_type": "valid",
        },
        label=-1,
    )
    valid_pos_wo_unseen = util.FilteredLabeledSplit(
        raw_split=valid_raw,
        key="valid_without_unseen",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "valid_without_unseen.del",
            "split_type": "valid",
        },
        label=1,
    )
    valid_neg_wo_unseen = util.FilteredLabeledSplit(
        raw_split=valid_raw,
        key="valid_without_unseen_negatives",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "valid_without_unseen_negatives.del",
            "split_type": "valid",
        },
        label=-1,
    )
    valid_raw.splits.extend(
        [valid_pos, valid_neg, valid_pos_wo_unseen, valid_neg_wo_unseen,]
    )

    # create splits: TEST
    test_pos = util.LabeledSplit(
        raw_split=test_raw,
        key="test",
        options={"type": "triples", "filename": "test.del", "split_type": "test"},
        label=1,
    )
    test_neg = util.LabeledSplit(
        raw_split=test_raw,
        key="test_negatives",
        options={
            "type": "triples",
            "filename": "test_negatives.del",
            "split_type": "test",
        },
        label=-1,
    )
    test_pos_wo_unseen = util.FilteredLabeledSplit(
        raw_split=test_raw,
        key="test_without_unseen",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "test_without_unseen.del",
            "split_type": "test",
        },
        label=1,
    )
    test_neg_wo_unseen = util.FilteredLabeledSplit(
        raw_split=test_raw,
        key="test_without_unseen_negatives",
        filter_with=train_raw,
        options={
            "type": "triples",
            "filename": "test_without_unseen_negatives.del",
            "split_type": "test",
        },
        label=-1,
    )
    test_raw.splits.extend(
        [test_pos, test_neg, test_pos_wo_unseen, test_neg_wo_unseen,]
    )

    # do the work
    util.process_splits(raw_dataset)
    util.update_string_files(raw_dataset, args)
    util.write_dataset_yaml(raw_dataset.config, args.folder)
