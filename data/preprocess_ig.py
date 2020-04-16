#!/usr/bin/env python
"""Preprocess a dataset of individual graphs (IG) into a the format expected by libkge.

Call as `preprocess_ig.py --folder <name>`. In the following the word "split" in a file-
name will refer to the file existing for each of the splits "train", "valid" and "test".
The original dataset should be stored in subfolder `name` and have files "annotations_split
.json". Each file containing a dictionary of IG names to lists of of triple-dictionaries,
each triple-dictionary must contain the elements: "subject", "predicate" and "object", where
"subject" and "object" must have elements: "category" and "bbox".

During preprocessing, each distinct entity name and each distinct distinct relation
name is assigned an index (dense). The index-to-object mapping is stored in files
"entity_map.del" and "relation_map.del", resp. For each IG for each unannotated sbj-
obj-pair a triple of the form (sbj, unknown, obj) is added. In the files "split_
index.del" the IGs are mapped to their names from "split_index.json".

IG datasets can be used in two ways, 1. based on a global graph and counts; 2. based
on the individual graphs. For 1. the triples (as indexes) are stored in "split_global.del"
and their corresponding counts are stored in a map file "split_counts.del". For 2. each
instance is assigned a global identifier inside its split which is mapped to its
IG in "split_instances.del" and to its class in "split_instance_classes.del". The triples
are stored twice, once in "split_triples_class.del" in the format (sbj_class_index,
pred_index, obj_class_index) and once in "split_triples_instance.del" in the format
(sbj_instance_index, pred_index, obj_instance_index). The triples are mapped to their
IGs in "split_triples_to_igs.del" (and to their counts per IG in "split_triples_to
_counts.del" for duplicate handling in evaluation).

Metadata information is stored in a file "dataset.yaml".
"""

import argparse
import yaml
import os.path
import json
from collections import OrderedDict, defaultdict
from ordered_set import OrderedSet


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

    splits = ["train", "valid", "test"]

    dataset_config = dict(
        name=args.folder,
        files=dict()
    )

    # raw triple files (of format as described above)
    raw_files = dict(zip(splits, ["annotations_" + split + ".json" for split in splits]))

    # mapping from indexes of igs to their names (ig_id, name)
    files_ig_names = dict(zip(splits, [split + "_index.del" for split in splits]))

    # global triples (sbj_class_id, rel_id, obj_class_id)
    files_global_triples = dict(zip(splits, [split + "_global.del" for split in splits]))

    # global triple to count mapping (triple_id, count)
    files_global_counts = dict(zip(splits, [split + "_counts.del" for split in splits]))

    # the following files are for working directly on individual graphs (IGs)

    # IG-level triples of the form (sbj_instance_id, rel_id, obj_instance_id)
    files_ig_triples_instance = dict(zip(splits, [split + "_ig_triples_instance.del" for split in splits]))

    # IG-level triples of the form (sbj_class_id, rel_id, obj_class_id)
    files_ig_triples_class = dict(zip(splits, [split + "_ig_triples_class.del" for split in splits]))

    # mapping from triples to their corresponding IG (triple_id, ig_id)
    files_ig_triples_to_igs = dict(zip(splits, [split + "_triples_to_igs.del" for split in splits]))

    # mapping from triples to their count (inside of one ig), for duplicate handling (triple_id, count)
    files_ig_triples_to_counts = dict(zip(splits, [split + "_ig_triples_to_counts.del" for split in splits]))

    # mapping from instances to their corresponding IG (instance_id, ig_id)
    files_ig_instances_to_igs = dict(zip(splits, [split + "_instances_to_igs.del" for split in splits]))

    # mapping from instances to their classes (instance_id, class_id)
    files_ig_instance_classes = dict(zip(splits, [split + "_instance_classes.del" for split in splits]))

    # TODO: Add instance to attribute mapping once Patrick has finished corresponding code
    # TODO: Add files and corresponding preprocessing for zero-shot triples

    # Build dataset config
    for split in splits:
        split_dict = {}
        global_dict = {}
        global_dict["triples"] = dict(filename=files_global_triples[split],
                                      type="triples", size=0)
        global_dict["counts"] = dict(filename=files_global_counts[split],
                                     type="map", key="triple", value="int")

        ig_dict = {}
        ig_dict["triples_instance"] = dict(filename=files_ig_triples_instance[split],
                                           type="triples", size=0)
        ig_dict["triples_class"] = dict(filename=files_ig_triples_class[split],
                                        type="triples", size=0)
        ig_dict["triples_to_igs"] = dict(filename=files_ig_triples_to_igs[split],
                                         type="map", key="triple", value="ig")
        ig_dict["triples_to_counts"] = dict(filename=files_ig_triples_to_counts[split],
                                            type="map", key="triple", value="count")
        ig_dict["instances_to_igs"] = dict(filename=files_ig_instances_to_igs[split],
                                           type="map", key="instance", value="ig")
        ig_dict["instance_classes"] = dict(filename=files_ig_instance_classes[split],
                                           type="map", key="instance", value="class")

        split_dict["global"] = global_dict
        split_dict["ig"] = ig_dict
        split_dict["ig_names"] = dict(filename=files_ig_names[split],
                                      type="map", key="ig", value="str")

        dataset_config["files"][split] = split_dict

    for obj in ["entity", "relation"]:
        dataset_config["files"][f"{obj}_ids"] = dict(filename=f"{obj}_ids.del",
                                                     type="map", key="int", value="str")

    S, P, O = 0, 1, 2
    S_n, P_n, O_n = "subject", "predicate", "object"
    category, bbox = "category", "bbox"

    # load raw data, split IG names from annotations (and write ig name mapping)
    data = {}
    for split in splits:
        with open(os.path.join(args.folder, raw_files[split]), "r") as f:
            data_raw = json.load(f)

        with open(os.path.join(args.folder, files_ig_names[split]), "w") as f:
            f.writelines(
                "\n".join(
                    [f"{i}\t{name}" for i, name in enumerate(data_raw.keys())]
                )
            )

        data[split] = list(data_raw.values())

    # build map of indexes to entities and relations in dataset and transform triples accordingly
    entities = {}
    relations = {}

    ent_id = 0
    rel_id = 0
    for split in splits:
        num_triples = 0
        for ig in data[split]:
            for t in ig:
                num_triples += 1
                if t[S_n][category] not in entities:
                    entities[t[S_n][category]] = ent_id
                    ent_id += 1
                if t[P_n] not in relations:
                    relations[t[P_n]] = rel_id
                    rel_id += 1
                if t[O_n][category] not in entities:
                    entities[t[O_n][category]] = ent_id
                    ent_id += 1

        for ig_index, ig in enumerate(data[split]):
            for t_index, t in enumerate(ig):
                transformed_triple = t
                transformed_triple[S_n][category] = entities[t[S_n][category]]
                transformed_triple[P_n] = relations[t[P_n]]
                transformed_triple[O_n][category] = entities[t[O_n][category]]
                transformed_triple[S_n][bbox] = t[S_n][bbox]
                transformed_triple[O_n][bbox] = t[O_n][bbox]

                data[split][ig_index][t_index] = transformed_triple

    # set unknown relation to last relation in map
    relations["unknown"] = rel_id
    print(f"{len(relations)} distinct relations")
    print(f"{len(entities)} distinct entities")
    print("Writing relation and entity map...")
    store_map(relations, os.path.join(args.folder, "relation_ids.del"))
    store_map(entities, os.path.join(args.folder, "entity_ids.del"))
    print("Done.")


    # generate sets of instances and triples (accounting for duplicates) per IG
    instances_per_ig = {}
    triples_per_ig = {}
    for split in splits:
        instances_per_ig[split] = {}
        triples_per_ig[split] = {}
        start_instance_id = 0
        for ig_index, ig in enumerate(data[split]):
            instances = OrderedSet()
            triples = defaultdict(int)
            existing_combinations = set()
            for t in ig:
                object_indices = []
                for pos in [S_n, O_n]:
                    object_ = [t[pos][category]]
                    object_.extend([t[pos][bbox][i] for i in range(3)])
                    object_ = tuple(object_)
                    instances.add(object_)
                    object_indices.append(instances.index(object_))
                existing_combinations.add(tuple(object_indices))
                triples[(
                    t[S_n][category], t[P_n], t[O_n][category],
                    object_indices[0] + start_instance_id,
                    object_indices[1] + start_instance_id
                )] += 1

            # add unannotated subject-object-pairs as triples with unknown relation
            for sbj_index in range(len(instances)):
                for obj_index in range(len(instances)):
                    if sbj_index != obj_index:
                        if not (sbj_index, obj_index) in existing_combinations:
                            sbj = instances[sbj_index]
                            obj = instances[obj_index]
                            triples[(
                                sbj[0], relations["unknown"], obj[0],
                                sbj_index + start_instance_id,
                                obj_index + start_instance_id
                            )] += 1

            instances_per_ig[split][ig_index] = list(instances)
            triples_per_ig[split][ig_index] = triples
            start_instance_id += len(instances)


    print("Writing triples per IG...")
    triples_of_classes = {}
    for split in splits:
        triples_of_classes[split] = []
        triples_of_instances = []
        triple_counts = []
        for ig_index, triples in triples_per_ig[split].items():
            for triple, count in triples.items():
                triples_of_classes[split].append((triple[0], triple[1], triple[2], ig_index))
                triples_of_instances.append((triple[3], triple[1], triple[4], ig_index))
                triple_counts.append(count)

        dataset_config["files"][split]["ig"]["triples_instance"]["size"] = len(triples_of_instances)
        dataset_config["files"][split]["ig"]["triples_class"]["size"] = len(triples_of_classes[split])

        with open(os.path.join(args.folder, files_ig_triples_class[split]), "w") as f_c,\
             open(os.path.join(args.folder, files_ig_triples_instance[split]), "w") as f_i,\
             open(os.path.join(args.folder, files_ig_triples_to_igs[split]), "w") as f_t,\
             open(os.path.join(args.folder, files_ig_triples_to_counts[split]), "w") as f_count:
            f_c.writelines(
                "\n".join(
                    [f"{t[S]}\t{t[P]}\t{t[O]}" for t in triples_of_classes[split]]
                )
            )
            f_i.writelines(
                "\n".join(
                    [f"{t[S]}\t{t[P]}\t{t[O]}" for t in triples_of_instances]
                )
            )
            f_t.writelines(
                "\n".join(
                    [f"{i}\t{t[3]}" for i, t in enumerate(triples_of_classes[split])]
                )
            )
            f_count.writelines(
                "\n".join(
                    [f"{i}\t{count}" for i, count in enumerate(triple_counts)]
                )
            )
    print("Done")

    print("Writing instance to class and to IG mapping...")
    for split in splits:
        with open(os.path.join(args.folder, files_ig_instance_classes[split]), "w") as f_c, \
             open(os.path.join(args.folder, files_ig_instances_to_igs[split]), "w") as f_i:
            start_instance_id = 0
            for ig_index in range(len(instances_per_ig[split])):
                for instance_index, instance in enumerate(instances_per_ig[split][ig_index]):
                    f_c.write(f"{instance_index + start_instance_id}\t{instance[0]}\n")
                    f_i.write(f"{instance_index + start_instance_id}\t{ig_index}\n")

                start_instance_id += len(instances_per_ig[split][ig_index])
    print("Done")

    print("Writing global triples and their counts...")
    for split in splits:
        triple_counts = defaultdict(int)
        for t in triples_of_classes[split]:
            triple_counts[(t[S], t[P], t[O])] += 1
        triple_counts = OrderedDict(triple_counts)

        with open(os.path.join(args.folder, files_global_triples[split]), "w") as f_t,\
             open(os.path.join(args.folder, files_global_counts[split]), "w") as f_c:
            f_t.writelines(
                "\n".join(
                    [f"{t[S]}\t{t[P]}\t{t[O]}" for t in triple_counts]
                )
            )
            f_c.writelines(
                "\n".join(
                    [f"{index}\t{c}" for index, c in enumerate(triple_counts.values())]
                )
            )

        dataset_config["files"][split]["global"]["triples"]["size"] = len(triple_counts)
    print("Done")

    print("Write dataset.yaml...")
    with open(os.path.join(args.folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))
    print("Done")
