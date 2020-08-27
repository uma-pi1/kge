import numpy as np
from os import path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Iterable
import yaml

### DEFAULT PREPROCESS UTILS ##########################################################


@dataclass
class Split:
    """Track data and meta-data of a dataset split.

    Attributes:
        file (str): File with tab-separated raw triples (can be labeled)
        collect_entities (bool): If true, entities contained in this split will be
            collected during preprocessing.
        collect_relations (bool): If true, relations contained in this split will be
            collected during preprocessing.
        SPO (dict): Mapping of subject, relation, object-index corresponding to the raw
            triples.
        derived_splits (list[DerivedSplitBase]): List of DerivedSplits, i.e, the final
            splits that ultimately will be written from this split.
        raw_data (list): List of triples represented with raw ids; created during
            preprocessing
        size (str): Number of triples; determined during preprocessing.
        entities (dict): Entities contained in this split if collected. Keys refer to
            raw id's and values to a dense index assigned during pre-processing.
        relations (dict): See entities.

    """

    # fields defined by user
    file: str
    collect_entities: bool = False
    collect_relations: bool = False
    SPO: Dict = None
    derived_splits: List[int] = field(default_factory=list)

    # fields defined during pre-processing
    raw_data: List[str] = None
    size: int = None
    entities: Dict = field(default_factory=dict)
    relations: Dict = field(default_factory=dict)

    def write_splits(self, entities: dict, relations: dict, folder):
        for derived_split in self.derived_splits:
            derived_split.prepare_process(folder)
        for n, t in enumerate(self.raw_data):
            for derived_split in self.derived_splits:
                derived_split.process_triple(t, entities, relations, n=n)
        for derived_split in self.derived_splits:
            derived_split.file.close()

    def update_config(self, config: Dict) -> Dict:
        """Update a dataset config with meta data of the derived splits."""
        for derived_split in self.derived_splits:
            for key, val in derived_split.options.items():
                config[f"files.{derived_split.key}.{key}"] = val
        return config


@dataclass
class DerivedSplitBase:
    """The final libKGE split derived and written from a base Split.

    Attributes:
        parent_split (Split): The parent split.
        key (str): The key in the dataset.yaml file.
        options (dict): Arbitrary dict of options. Key_ value_ pairs will be added to
            to dataset.yaml under the respective key entry of the split.

    """

    parent_split: Split = None
    key: str = None
    options: Dict = None

    def prepare_process(self, folder):
        self.file = open(path.join(folder, self.options["filename"]), "w")
        self.options["size"] = 0

    def process_triple(self, triple, entities, relations, **kwargs):
        write_triple(
            self.file,
            entities,
            relations,
            triple,
            self.parent_split.SPO["S"],
            self.parent_split.SPO["P"],
            self.parent_split.SPO["O"],
        )
        self.options["size"] += 1


@dataclass
class DerivedSplitFiltered(DerivedSplitBase):
    """A filtered derived split.

    Attributes:
        filter_with (Split): The Split of which entities and relations shall be taken
            for filtering. The derived split exclusively contains triples where
            entities and relations are known in the filter_with Split.
    """

    filter_with: Split = None

    def process_triple(self, triple, entities, relations, **kwargs):
        S, P, O = (
            self.parent_split.SPO["S"],
            self.parent_split.SPO["P"],
            self.parent_split.SPO["O"],
        )
        if (
            triple[S] in self.filter_with.entities
            and triple[O] in self.filter_with.entities
            and triple[P] in self.filter_with.relations
        ):
            write_triple(self.file, entities, relations, triple, S, P, O)
            self.options["size"] += 1


@dataclass
class DerivedSplitSample(DerivedSplitBase):
    """A derived sub-sample Split.

       Attributes:
           sample_size (int): Size of the subsample.
           sample (Iterable[int]): Randomly selected triple indexes with size
               sample_size; determined in  prepare_process().

    """

    sample_size: int = None
    sample: Iterable[int] = None

    def prepare_process(self, folder):
        super().prepare_process(folder)
        self.sample = np.random.choice(
            len(self.parent_split.raw_data), self.sample_size, False
        )

    def process_triple(self, triple, entities, relations, **kwargs):
        if kwargs["n"] in self.sample:
            write_triple(
                self.file,
                entities,
                relations,
                triple,
                self.parent_split.SPO["S"],
                self.parent_split.SPO["P"],
                self.parent_split.SPO["O"],
            )
            self.options["size"] += 1


@dataclass
class RawDataset:
    """A raw relational dataset.

     Contains the RawSplits of the dataset to be processed and the final config;
     is generated automatically in analyze_splits().

     Attributes:
         splits (list[Split]): List of Splits.
         all_entities (dict): Distinct entities over all splits. Keys refer to
            raw entity-id's and values to the dense index.
         all_relations (dict): See all entities.
         config (dict): Raw dictionary holding the dataset config options.
         folder (str): Path to the dataset folder.

     """

    splits: List[Split]
    all_entities: Dict[str, int]
    all_relations: Dict[str, int]
    config: Dict[str, str]
    folder: str


def process_splits(dataset: RawDataset):
    for split in dataset.splits:
        split.write_splits(
            entities=dataset.all_entities,
            relations=dataset.all_relations,
            folder=dataset.folder,
        )
        # add the collected meta data to the config
        split.update_config(dataset.config)


def analyze_splits(splits: List[Split], folder: str) -> RawDataset:
    """Read a collection of raw splits and create a RawDataset.

    Args:
        splits (list[Splits]): List of RawSplits.
        folder (str): Folder of the dataset containing the files.
    """
    # read data and collect entities and relations
    all_entities = {}
    all_relations = {}
    ent_id = 0
    rel_id = 0
    for split in splits:
        with open(path.join(folder, split.file), "r") as f:
            split.raw_data = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            S, P, O = split.SPO["S"], split.SPO["P"], split.SPO["O"]
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
        splits=splits,
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
class DerivedLabeledSplit(DerivedSplitBase):
    label: int = None

    def process_triple(self, triple, entities, relations, **kwargs):
        if int(triple[3]) == self.label:
            write_triple(
                self.file,
                entities,
                relations,
                triple,
                self.parent_split.SPO["S"],
                self.parent_split.SPO["P"],
                self.parent_split.SPO["O"],
            )
            self.options["size"] += 1


@dataclass
class DerivedLabeledSplitFiltered(DerivedSplitFiltered):
    label: int = None

    def process_triple(self, triple, entities, relations, **kwargs):
        S, P, O = (
            self.parent_split.SPO["S"],
            self.parent_split.SPO["P"],
            self.parent_split.SPO["O"],
        )
        if int(triple[3]) == self.label and (
            triple[S] in self.filter_with.entities
            and triple[O] in self.filter_with.entities
            and triple[P] in self.filter_with.relations
        ):
            write_triple(self.file, entities, relations, triple, S, P, O)
            self.options["size"] += 1
