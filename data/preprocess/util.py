import numpy as np


def store_map(symbol_map: dict, filename: str):
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


def process_raw_split_files(raw_split_files: dict, folder: str,  order_sop: bool = False):
    """Read a collection of raw split files and collect meta data."""

    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # read data and collect entities and relations
    split_sizes = {}
    raw = {}
    entities = {}
    relations = {}
    entities_in_train = {}
    relations_in_train = {}
    ent_id = 0
    rel_id = 0
    for split, filename in raw_split_files.items():
        with open(folder + "/" + filename, "r") as f:
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

    return split_sizes, raw,  entities, relations, entities_in_train, relations_in_train


def write_triple(f, ent, rel, t, S, P, O):
    f.write(str(ent[t[S]]) + "\t" + str(rel[t[P]]) + "\t" + str(ent[t[O]]) + "\n")


def process_split(
        entities: dict,
        relations: dict,
        file: str,
        raw_split: list,
        order_sop: bool = False,
        create_sample: bool = False,
        create_filtered: bool = False,
        **kwargs
):
    """From a raw split, write a split file with indexes.

     Optionally, a filtered split file and a sample of the raw split can be created.

     """
    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # the sampled split is a randomly chosen subsample of raw_split
    if create_sample:
        sample_size = kwargs["sample_size"]
        sample_file = kwargs["sample_file"]
        sample = np.random.choice(len(raw_split), sample_size, False)
        sample_f = open(sample_file, "w")

    # the filtered split cosists of triples from the raw split where both entities
    # and relation exist in filter_entities/filter_relation
    if create_filtered:
        filtered_size = 0
        filter_entities = kwargs["filter_entities"]
        filter_relations = kwargs["filter_relations"]
        filter_f = open(kwargs["filtered_file"], "w")

    with open(file, "w") as f:
        for n, t in enumerate(raw_split):
            write_triple(f, entities, relations, t, S, P, O)
            if create_sample and n in sample:
                write_triple(sample_f, entities, relations, t, S, P, O)
            if create_filtered and t[S] in filter_entities \
                               and t[O] in filter_entities \
                               and t[P] in filter_relations:
                write_triple(filter_f, entities, relations, t, S, P, O)
                filtered_size += 1
        if create_filtered:
            return filtered_size


def process_pos_neg_split(
        entities: dict,
        relations: dict,
        pos_file: str,
        neg_file: str,
        raw_split: list,
        order_sop: bool = False,
        create_filtered: bool = False,
        **kwargs
):
    """From a raw split containing labeled triples, write split files with indexes.

     Optionally, filtered split files can be created.

     """
    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # the filtered split cosists of triples from the raw split where both entities
    # and relation exist in filter_entities/filter_relation
    if create_filtered:
        filter_entities = kwargs["filter_entities"]
        filter_relations = kwargs["filter_relations"]
        filter_pos_f = open(kwargs["filtered_pos_file"], "w")
        filter_neg_f = open(kwargs["filtered_neg_file"], "w")
        filtered_pos_size = 0
        filtered_neg_size = 0

    pos_size = 0
    neg_size = 0
    with open(pos_file, "w") as pos_file, \
         open(neg_file, "w") as neg_file:

        for n, t in enumerate(raw_split):
            if int(t[3]) == -1:
                file_wrapper = neg_file
                filtered_file_wrapper = filter_neg_f
                neg_size += 1
            else:
                file_wrapper = pos_file
                filtered_file_wrapper = filter_pos_f
                pos_size += 1
            write_triple(file_wrapper, entities, relations, t, S, P, O)
            if create_filtered and t[S] in filter_entities \
                               and t[O] in filter_entities \
                               and t[P] in filter_relations:

                if int(t[3]) == -1:
                    filtered_neg_size += 1
                else:
                    filtered_pos_size += 1
                write_triple(filtered_file_wrapper, entities, relations, t, S, P, O)

        if create_filtered:
            return pos_size, filtered_pos_size, neg_size, filtered_neg_size
        else:
            return pos_size, neg_size


def write_obj_meta(config: dict):
    for obj in ["entity", "relation"]:
        config[f"files.{obj}_ids.filename"] = f"{obj}_ids.del"
        config[f"files.{obj}_ids.type"] = "map"


def write_split_meta(split_types: list, config: dict, split_sizes: dict):
    for split_type in split_types:
        for raw_split, split_dict in split_type.items():
            file_key = split_dict["file_key"]
            file_name = split_dict["file_name"]
            config[f"files.{file_key}.filename"] = file_name
            config[f"files.{file_key}.type"] = "triples"
            config[f"files.{file_key}.size"] = split_sizes[file_key]

