import numpy as np
from os import path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import yaml


@dataclass
class RawSplit:
    """A raw split.

    Attributes:
        name (str): Name of the raw split
        file (str): File with tab-separated raw triples (can be labeled)
        raw_data (list): List with triples encoded by raw id's
        size (str): Number of triples.
        entities (dict): Entities contained in this split. Keys refer to raw
            id's and values to a dense index assigned during pre-processing.
        relations (dict): See entities.
        create_filtered (bool): If true, an additional filtered split will be created.
        filter_with (RawSplit): Dictionary mapping with entities that should be included.
        meta: (dict): Additional entries that this split adds to its config entries.

    """
    # fields defined by user
    name: str
    file: str
    create_filtered: bool
    filter_with = None
    create_sample: bool
    sample_size: int = None
    meta: Dict = None

    # fields defined during pre-processing
    raw_data: List[str] = None
    size: int = None
    entities: Dict = field(default_factory=dict)
    relations: Dict = field(default_factory=dict)
    filtered_size: int = None
    create_sample: bool = False



@dataclass
class RawDataset:
    """A raw relational dataset.

     Contains raw data for the splits of a dataset and various types of meta data. Is
     created automatically in analyze_raw_splits().

     Attributes:
         raw_splits (list[RawSplit]): List of RawSplits.
         all_entities (dict): Distinct entities over all splits. Keys refer to
            raw entity-id's and values to the dense index.
         all_relations (dict): See all entities.
         config (dict): Raw dictionary holding the dataset config options.
         folder (str): Path to the dataset folder.

     """
    raw_splits: List[RawSplit]
    all_entities: Dict[str, int]
    all_relations: Dict[str, int]
    config: Dict[str, str]
    folder: str
    order_sop: bool = False


def analyze_raw_splits(
        raw_splits: List[RawSplit],
        folder: str,
        order_sop: bool = False,

) -> RawDataset:
    """Read a collection of raw splits and create a RawDataset.

    Args:
        raw_splits (list[RawSplits]): List of RawSplits.
        folder (str): Folder of the dataset containing the files.
        order_sop (bool): True when the order in the raw triples is (S,O,P) instead of
            (S,P,O).     
    """

    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # read data and collect entities and relations
    all_entities = {}
    all_relations = {}
    ent_id = 0
    rel_id = 0
    for split in raw_splits:
        with open(path.join(folder, split.file), "r") as f:
            split.raw_data = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            for t in split.raw_data:
                if t[S] not in all_entities:
                    all_entities[t[S]] = ent_id
                    ent_id += 1
                if t[P] not in all_relations:
                    all_relations[t[P]] = rel_id
                    rel_id += 1
                if t[O] not in all_entities:
                    all_entities[t[O]] = ent_id
                    ent_id += 1
                # TODO you could use a flag for a raw split if it is desired to save
                #  its objects separately, e.g. only for the train split it is needed
                #  for the current dataset
                split.entities[t[S]] = all_entities[t[S]]
                split.entities[t[O]] = all_entities[t[O]]
                split.relations[t[P]] = all_relations[t[P]]
            split.size = len(split.raw_data)
            print(
                f"Found {split.size} triples in {split.name} split "
                f"(file: {split.file})."
            )
    print(f"{len(all_relations)} distinct relations")
    print(f"{len(all_entities)} distinct entities")

    config = dict(
        name=folder, num_entities=len(all_entities), num_relations=len(all_relations),
    )

    dataset = RawDataset(
            raw_splits=raw_splits,
            all_entities=all_entities,
            all_relations=all_relations,
            config=config,
            folder=folder,
            order_sop=order_sop,
    )

    # write entity/relation maps and update config
    process_obj_meta(dataset)

    return dataset


def process_obj_meta(dataset: RawDataset):
    """Write entity and relation maps and update config with respective keys."""
    print("Writing relation and entity map...")
    store_map(dataset.all_relations, path.join(dataset.folder, "relation_ids.del"))
    store_map(dataset.all_entities, path.join(dataset.folder, "entity_ids.del"))
    for obj in ["entity", "relation"]:
        dataset.config[f"files.{obj}_ids.filename"] = f"{obj}_ids.del"
        dataset.config[f"files.{obj}_ids.type"] = "map"


def store_map(symbol_map: dict, filename: str):
    """Write a map file."""
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


def write_triple(f, ent, rel, t, S, P, O):
    """Write a triple to a file. """
    f.write(str(ent[t[S]]) + "\t" + str(rel[t[P]]) + "\t" + str(ent[t[O]]) + "\n")


def write_split(
        raw_triples: List[List],
        file_name: str,
        entities: dict,
        relations: dict,
        order_sop: bool = False,
        create_sample: bool = False,
        sample_size: int = None,
        sample_file: str = None,
        create_filtered: bool = False,
        filtered_file: str = None,
        filtered_include_ent: Union[dict, list] = None,
        filtered_include_rel: Union[dict, list] = None,
):
    """From a raw split, write a split file using indexes.

     Optionally, a filtered split file and a sample of the raw split can be created.

     Args:
         triples (list): List of triples.
         file_name (str): File name where the split is written to.
         entities (dict): mapping of all entities
         relations (dict): mapping of all relations
         order_sop (bool): True when the order of the triples in "dataset" is (S,O,P)
         create_sample (bool): Write an additional random subset of the split.
         sample_size (int): Size of the subset.
         sample_file (str): Filename of the subset.
         create_filtered(bool): Write an additional filtered version of the split.
         filtered_file (str): Filename of the filtered split.
         filtered_include_ent : Dict or List. Triples containing these entities will be
         included in the filtered split.
         filtered_include_rel: see filtered_include_ent.

     """
    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    # the sampled split is a randomly chosen subsample
    if create_sample:
        sample = np.random.choice(len(raw_triples), sample_size, False)
        sample_f = open(path.join(sample_file), "w")

    # the filtered split consists of triples from the raw split where both entities
    # and relation exist in filtered_include_ent/filter_relation
    if create_filtered:
        filtered_size = 0
        filter_f = open(filtered_file, "w")

    with open(file_name, "w") as f:
        for n, t in enumerate(raw_triples):
            write_triple(f, entities, relations, t, S, P, O)
            if create_sample and n in sample:
                write_triple(sample_f, entities, relations, t, S, P, O)
            if create_filtered and t[S] in filtered_include_ent \
                    and t[O] in filtered_include_ent \
                    and t[P] in filtered_include_rel:
                write_triple(filter_f, entities, relations, t, S, P, O)
                filtered_size += 1


  # write to config everything you have done
    dataset.config[f"files.{file_key}.size"] = dataset.raw_split_sizes[split]
    if create_filtered:
        dataset.config[f"files.{filtered_key}.size"] = filtered_size
    if create_sample:
        dataset.config[f"files.{sample_key}.size"] = sample_size


def process_dataset_splits(raw_dataset: RawDataset):
    for raw_split in raw_dataset.raw_splits:
        write_split(
            raw_triples=raw_split.raw_data,
            file_name=path.join(raw_dataset.folder, f"{raw_split.file}.del"),
            entitie=raw_dataset.all_entities,
            relations=raw_dataset.all_relations,
            order_sop=raw_dataset.order_sop,
            create_sample=raw_split.create_sample,
            sample_size=raw_split.sample_size,
            sample_file=path.join(raw_dataset.folder, f"{raw_split.file}_sample.del"),
            create_filtered=raw_split.create_filtered
            filtered_file=

        )


 raw_triples: List[List],
        file_name: str,
        entities: dict,
        relations: dict,
        order_sop: bool = False,
        create_sample: bool = False,
        sample_size: int = None,
        sample_file: str = None,
        create_filtered: bool = False,
        filtered_file: str = None,
        filtered_include_ent: Union[dict, list] = None,
        filtered_include_rel: Union[dict, list] = None,



def process_split(
        split: str,
        dataset: RawDataset,
        file_name: str,
        file_key: str,
        order_sop: bool = False,
        create_sample: bool = False,
        sample_size: int = None,
        sample_file: str = None,
        sample_key: str = None,
        create_filtered: bool = False,
        filtered_file: str = None,
        filtered_key: str = None,
        filtered_include_ent: Union[dict, list] = None,
        filtered_include_rel: Union[dict, list] = None,
):
    """From a raw split, write a split file using indexes.

     Optionally, a filtered split file and a sample of the raw split can be created.
     
     Args:
         split (str): Name of the raw split.
         dataset (RawDataset): The RawDataset containing raw splits.
         file_name (str): File name where the split is written to.
         file_key (str): Key used in the dataset config.
         order_sop (bool): True when the order of the triples in "dataset" is (S,O,P)
         create_sample (bool): Write an additional random subset of the split.
         sample_size (int): Size of the subset.
         sample_file (str): Filename of the subset.
         sample_key (str): Key used for the subset in the dataset config.
         create_filtered(bool): Write an additional filtered version of the split.
         filtered_file (str): Filename of the filtered split.
         filtered_key (str): Key used in the dataset config for the filtered split.
         filtered_include_ent : Dict or List. Triples containing these entities will be 
         included in the filtered split.
         filtered_include_rel: see filtered_include_ent.       

     """
    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    raw_split = dataset.raw_split_data[split]
    entities = dataset.all_entities
    relations = dataset.all_relations

    # the sampled split is a randomly chosen subsample of raw_split
    if create_sample:
        sample = np.random.choice(len(raw_split), sample_size, False)
        sample_f = open(path.join(dataset.folder, sample_file), "w")

    # the filtered split cosists of triples from the raw split where both entities
    # and relation exist in filtered_include_ent/filter_relation
    if create_filtered:
        filtered_size = 0
        filter_f = open(path.join(dataset.folder, filtered_file), "w")

    print(f"Writing {split} triples...")

    file = path.join(dataset.folder, file_name)
    with open(file, "w") as f:
        for n, t in enumerate(raw_split):
            write_triple(f, entities, relations, t, S, P, O)
            if create_sample and n in sample:
                write_triple(sample_f, entities, relations, t, S, P, O)
            if create_filtered and t[S] in filtered_include_ent \
                               and t[O] in filtered_include_ent \
                               and t[P] in filtered_include_rel:
                write_triple(filter_f, entities, relations, t, S, P, O)
                filtered_size += 1

    # write to config everything you have done
    dataset.config[f"files.{file_key}.size"] = dataset.raw_split_sizes[split]
    if create_filtered:
        dataset.config[f"files.{filtered_key}.size"] = filtered_size
    if create_sample:
        dataset.config[f"files.{sample_key}.size"] = sample_size


def process_pos_neg_split(
        split,
        dataset: RawDataset,
        pos_file: str,
        pos_key: str,
        neg_file: str,
        neg_key: str,
        order_sop: bool = False,
        create_filtered: bool = False,
        filtered_include_ent: dict = None,
        filtered_include_rel: dict = None,
        filtered_pos_file: str = None,
        filtered_pos_key: str = None,
        filtered_neg_file: str = None,
        filtered_neg_key: str = None
):
    """From a raw split containing labeled triples, write split files with indexes.

     Optionally, filtered split files can be created.

     See process_split().

     """
    if order_sop:
        S, P, O = 0, 2, 1
    else:
        S, P, O = 0, 1, 2

    raw_split = dataset.raw_split_data[split]
    entities = dataset.all_entities
    relations = dataset.all_relations

    # the filtered split cosists of triples from the raw split where both entities
    # and relation exist in filtered_include_ent/filter_relation
    if create_filtered:
        filter_pos_f = open(path.join(dataset.folder, filtered_pos_file), "w")
        filter_neg_f = open(path.join(dataset.folder, filtered_neg_file), "w")
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
            if create_filtered and t[S] in filtered_include_ent \
                               and t[O] in filtered_include_ent \
                               and t[P] in filtered_include_rel:

                if int(t[3]) == -1:
                    filtered_neg_size += 1
                else:
                    filtered_pos_size += 1
                write_triple(filtered_file_wrapper, entities, relations, t, S, P, O)

        # update dataset config with everything you have done
        dataset.config[f"files.{pos_key}.size"] = pos_size
        dataset.config[f"files.{neg_key}.size"] = neg_size

        if create_filtered:
            dataset.config[f"files.{filtered_pos_key}.size"] = filtered_pos_size
            dataset.config[f"files.{filtered_neg_key}.size"] = filtered_neg_size


def write_split_meta(split_types: list, config: dict):
    """Update a config dict with meta data from split types. """
    for split_type in split_types:
        for raw_split, split_dict in split_type.items():
            file_key = split_dict["file_key"]
            file_name = split_dict["file_name"]
            config[f"files.{file_key}.filename"] = file_name
            config[f"files.{file_key}.type"] = "triples"


def write_dataset_config(config: dict, folder: str):
    """Write a dataset.yaml file given a config dictionary and a folder path. """
    print(yaml.dump(dict(dataset=config)))
    with open(path.join(folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=config)))
