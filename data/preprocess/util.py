import numpy as np
from os import path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import yaml

### DEFAULT PREPROCESS UTILS ##########################################################


@dataclass
class RawSplitBase:
    """Base RawSplit, tracks data and meta data of a raw split.

    Attributes:
        file (str): File with tab-separated raw triples (can be labeled)
        raw_data (list): List of triples encoded with raw ids.
        size (str): Number of triples; determined during preprocessing.
        collect_entities (bool): If true, entities contained in this split will be
            collected during preprocessing.
        collect_relations (bool): When true, relations contained in this split will be
            collected during preprocessing.
        entities (dict): Entities contained in this split when collected. Keys refer to
            raw id's and values to a dense index assigned during pre-processing.
        relations (dict): See entities.

    """

    # fields defined by user
    file: str
    # fields defined during pre-processing
    raw_data: List[str] = None
    order_sop: bool = False
    size: int = None
    collect_entities: bool = False
    collect_relations: bool = False
    entities: Dict = field(default_factory=dict)
    relations: Dict = field(default_factory=dict)

    def write_splits(self, entities: Dict, relations: Dict, folder: str):
        """Write the derived splits of this RawSplit and collect meta-data. """
        raise NotImplemented

    def update_config(self, config: Dict) -> Dict:
        """Update a dataset config with meta data of the derived splits."""
        raise NotImplemented


@dataclass
class RawSplit(RawSplitBase):
    """A raw split, tracks meta data and info about splits to be derived.

    Attributes:
        derived_split_key (str): The config key of the default split derived from
            the raw split.
        derived_split_options (dict): A dictionary with config options that will
            be added to the config entry of the derived default split.
        derived_sample_split_key (str): When a key is given, an additional subsample
            with config entry of this key will be derived from the raw split.
        sample_split_options (dict): Dict with options to be added to the config
            entry of the subsample split.
        sample_size (int): Size of the subsample split.
        derived_filtered_split_key (str): When a key is given, an additional filtered
            split will be derived from the raw split.
        filter_with (RawSplit): A RawSplit, entities and relations from this split
            are used as filter for the filtered split.
        filtered_split_options (str): See sample split options.


    """

    # fields defined by user
    derived_split_key: str = None
    derived_split_options: Dict = None

    derived_sample_split_key: str = None
    sample_size: int = None
    sample_split_options: Dict = None

    derived_filtered_split_key: str = None
    filter_with: RawSplitBase = None
    filtered_split_options: Dict = None

    def write_splits(self, entities: dict, relations: dict, folder: str):
        """Write the derived splits of this RawSplit and collect meta-data. """
        if self.order_sop:
            S, P, O = 0, 2, 1
        else:
            S, P, O = 0, 1, 2

        # the sampled split is a randomly chosen subsample
        if self.derived_sample_split_key:
            sample = np.random.choice(len(self.raw_data), self.sample_size, False)
            sample_f = open(
                path.join(folder, self.sample_split_options["filename"]), "w"
            )

        # the filtered split consists of triples from the raw split where both entities
        # and relation exist in filtered_include_ent/filter_relation
        if self.derived_filtered_split_key:
            filtered_size = 0
            filter_f = open(
                path.join(folder, self.filtered_split_options["filename"]), "w")

        filename = path.join(folder, self.derived_split_options["filename"])
        with open(filename, "w") as f:
            for n, t in enumerate(self.raw_data):
                write_triple(f, entities, relations, t, S, P, O)
                if self.derived_sample_split_key and n in sample:
                    write_triple(sample_f, entities, relations, t, S, P, O)
                if (
                    self.derived_filtered_split_key
                    and t[S] in self.filter_with.entities
                    and t[O] in self.filter_with.entities
                    and t[P] in self.filter_with.relations
                ):
                    write_triple(filter_f, entities, relations, t, S, P, O)
                    filtered_size += 1

        # collect meta data of everything you have done
        self.derived_split_options["size"] = self.size
        if self.derived_filtered_split_key:
            self.filtered_split_options["size"] = filtered_size
        if self.derived_sample_split_key:
            self.sample_split_options["size"] = self.sample_size

    def update_config(self, config: Dict) -> Dict:
        """Update a dataset config with meta data of the derived splits."""
        for key, val in self.derived_split_options.items():
            config[f"files.{self.derived_split_key}.{key}"] = val
        if self.derived_sample_split_key:
            for key, val in self.sample_split_options.items():
                config[f"files.{self.derived_sample_split_key}.{key}"] = val
        if self.derived_filtered_split_key:
            for key, val in self.filtered_split_options.items():
                config[f"files.{self.derived_filtered_split_key}.{key}"] = val
        return config


@dataclass
class RawDataset:
    """A raw relational dataset.

     Contains the RawSplits of the dataset to be processed and the final config;
     is generated automatically in analyze_raw_splits().

     Attributes:
         raw_splits (list[RawSplit]): List of RawSplits.
         all_entities (dict): Distinct entities over all splits. Keys refer to
            raw entity-id's and values to the dense index.
         all_relations (dict): See all entities.
         config (dict): Raw dictionary holding the dataset config options.
         folder (str): Path to the dataset folder.

     """

    raw_splits: List[RawSplitBase]
    all_entities: Dict[str, int]
    all_relations: Dict[str, int]
    config: Dict[str, str]
    folder: str


def process_splits(dataset: RawDataset):
    for raw_split in dataset.raw_splits:
        raw_split.write_splits(
            entities=dataset.all_entities,
            relations=dataset.all_relations,
            folder=dataset.folder,
        )
        # add the collected meta data to the config
        raw_split.update_config(dataset.config)


def analyze_raw_splits(
    raw_splits: List[RawSplit], folder: str, order_sop: bool = False,
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
                if split.collect_entities:
                    split.entities[t[S]] = all_entities[t[S]]
                    split.entities[t[O]] = all_entities[t[O]]
                if split.collect_relations:
                    split.relations[t[P]] = all_relations[t[P]]
            split.size = len(split.raw_data)
            print(f"Found {split.size} triples in {split.file}")
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
    )

    # write entity/relation maps and update config
    write_obj_maps(dataset)

    return dataset


def write_obj_maps(dataset: RawDataset):
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


def write_dataset_config(config: dict, folder: str):
    """Write a dataset.yaml file given a config dictionary and a folder path. """
    print(yaml.dump(dict(dataset=config)))
    with open(path.join(folder, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=config)))


### WN11 UTILS #########################################################################


@dataclass
class PosNegRawSplit(RawSplitBase):
    """A raw split containing positive and negative triples."""

    # fields defined by user
    derived_split_pos_key: str = None
    derived_split_pos_options: Dict = None

    derived_split_neg_key: str = None
    derived_split_neg_options: Dict = None

    derived_split_filtered_pos_key: str = None
    filtered_split_pos_options: Dict = None

    derived_split_filtered_neg_key: str = None
    filtered_split_neg_options: Dict = None

    filter_with: RawSplitBase = None

    def write_splits(self, entities: dict, relations: dict, folder: str):
        if self.order_sop:
            S, P, O = 0, 2, 1
        else:
            S, P, O = 0, 1, 2

        # the filtered split cosists of triples from the raw split where both
        create_filtered = (
            self.derived_split_filtered_neg_key and self.derived_split_filtered_pos_key
        )
        if create_filtered:
            filter_pos_f = open(
                path.join(folder, self.filtered_split_pos_options["filename"]), "w"
            )
            filter_neg_f = open(
                path.join(folder, self.filtered_split_neg_options["filename"]), "w"
            )
            filtered_pos_size = 0
            filtered_neg_size = 0

        pos_size = 0
        neg_size = 0
        pos_file = path.join(folder, self.derived_split_pos_options["filename"])
        neg_file = path.join(folder, self.derived_split_neg_options["filename"])
        with open(pos_file, "w") as pos_file, open(neg_file, "w") as neg_file:
            for n, t in enumerate(self.raw_data):
                if int(t[3]) == -1:
                    file_wrapper = neg_file
                    neg_size += 1
                else:
                    file_wrapper = pos_file
                    pos_size += 1
                write_triple(file_wrapper, entities, relations, t, S, P, O)
                if (
                    create_filtered
                    and t[S] in self.filter_with.entities
                    and t[O] in self.filter_with.entities
                    and t[P] in self.filter_with.relations
                ):

                    if int(t[3]) == -1:
                        filtered_neg_size += 1
                        filtered_file_wrapper = filter_neg_f
                    else:
                        filtered_pos_size += 1
                        filtered_file_wrapper = filter_pos_f
                    write_triple(filtered_file_wrapper, entities, relations, t, S, P, O)

            # collect meta data of everything you have done
            self.derived_split_pos_options["size"] = pos_size
            self.derived_split_neg_options["size"] = neg_size
            if create_filtered:
                self.filtered_split_pos_options["size"] = filtered_pos_size
                self.filtered_split_neg_options["size"] = filtered_neg_size

    def update_config(self, config: Dict) -> Dict:
        """Update a dataset config with meta data of the derived splits."""

        options = [self.derived_split_pos_options, self.derived_split_neg_options]

        file_keys = [
            self.derived_split_pos_key,
            self.derived_split_neg_key,
        ]
        filt_pos_key = self.derived_split_filtered_pos_key
        filt_neg_key = self.derived_split_filtered_neg_key
        if filt_pos_key and filt_neg_key:
            options.extend(
                [self.filtered_split_pos_options, self.filtered_split_neg_options]
            )
            file_keys.extend([filt_pos_key, filt_neg_key])
        for file_key, options in zip(file_keys, options):
            for key, val in options.items():
                config[f"files.{file_key}.{key}"] = val
        return config
