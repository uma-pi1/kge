#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one SPO triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_map.del" and "relation_map.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Metadata information is stored in a file
"dataset.yaml".

"""

import argparse
import yaml
import os.path
import numpy as np

def store_map(symbol_map, filename):
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--order_sop", action="store_true")
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")
    raw_split_files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
    split_files = {"train": "train.del", "valid": "valid.del", "test": "test.del"}
    string_files = {"entity_strings": "entity_strings.del", "relation_strings": "relation_strings.del"}
    split_files_without_unseen = {"train_sample": "train_sample.del", "valid_without_unseen": "valid_without_unseen.del", 
            "test_without_unseen": "test_without_unseen.del"}
    split_sizes = {}

    if args.order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # read data and collect entities and relations
    raw = {}
    entities = {}
    relations = {}
    entities_in_train = {}
    relations_in_train = {}
    ent_id = 0
    rel_id = 0
    for split, filename in raw_split_files.items():
        with open(args.folder + "/" + filename, "r") as f:
            raw[split] = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            for t in raw[split]:
                if t[S] not in entities:
                    entities[t[S]] = ent_id
                    ent_id += 1
                if t[P] not in relations:
                    relations[t[P]] = rel_id
                    rel_id += 1
                if t[O] not in entities:
                    entities[t[O]] = ent_id
                    ent_id += 1
            print(
                f"Found {len(raw[split])} triples in {split} split "
                f"(file: {filename})."
            )
            split_sizes[split] = len(raw[split])
            if "train" in split:
                entities_in_train = entities.copy()
                relations_in_train = relations.copy()
    
    print(f"{len(relations)} distinct relations")
    print(f"{len(entities)} distinct entities")
    print("Writing relation and entity map...")
    store_map(relations, os.path.join(args.folder, "relation_ids.del"))
    store_map(entities, os.path.join(args.folder, "entity_ids.del"))
    print("Done.")

    # write out triples using indexes
    print("Writing triples...")
    without_unseen_sizes = {}
    for split, filename in split_files.items():
        if split in ["valid", "test"]:
            split_without_unseen = split + "_without_unseen"
            f_wo_unseen = open(os.path.join(args.folder, 
                                split_files_without_unseen[split_without_unseen]), "w")
        else:
            split_without_unseen = split + "_sample"
            f_tr_sample = open(os.path.join(args.folder, 
                                split_files_without_unseen[split_without_unseen]), "w")
            train_sample = np.random.choice(split_sizes["train"], split_sizes["valid"], False)
        with open(os.path.join(args.folder, filename), "w") as f:
            size_unseen = 0
            for n, t in enumerate(raw[split]):
                f.write(
                    str(entities[t[S]])
                    + "\t"
                    + str(relations[t[P]])
                    + "\t"
                    + str(entities[t[O]])
                    + "\n"
                )
                if split == "train" and n in train_sample:
                    f_tr_sample.write(
                        str(entities[t[S]])
                        + "\t"
                        + str(relations[t[P]])
                        + "\t"
                        + str(entities[t[O]])
                        + "\n"
                    )
                    size_unseen += 1
                elif split in ["valid", "test"] and t[S] in entities_in_train and \
                    t[O] in entities_in_train and t[P] in relations_in_train:
                    f_wo_unseen.write(
                        str(entities[t[S]])
                        + "\t"
                        + str(relations[t[P]])
                        + "\t"
                        + str(entities[t[O]])
                        + "\n"
                    )
                    size_unseen += 1
            without_unseen_sizes[split_without_unseen] = size_unseen

    # write config
    print("Writing dataset.yaml...")
    dataset_config = dict(
        name=args.folder,
        num_entities=len(entities),
        num_relations=len(relations),
    )
    for obj in [ "entity", "relation" ]:
        dataset_config[f"files.{obj}_ids.filename"] = f"{obj}_ids.del"
        dataset_config[f"files.{obj}_ids.type"] = "map"
    for split in split_files.keys():
        dataset_config[f"files.{split}.filename"] = split_files.get(split)
        dataset_config[f"files.{split}.type"] = "triples"
        dataset_config[f"files.{split}.size"] = split_sizes.get(split)
    for split in split_files_without_unseen.keys():
        dataset_config[f"files.{split}.filename"] = split_files_without_unseen.get(split)
        dataset_config[f"files.{split}.type"] = "triples"
        dataset_config[f"files.{split}.size"] = without_unseen_sizes.get(split)
    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            dataset_config[f"files.{string}.filename"] = string_files.get(string)
            dataset_config[f"files.{string}.type"] = "idmap"
    print(yaml.dump(dict(dataset=dataset_config)))
    with open(os.path.join(args.folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))
