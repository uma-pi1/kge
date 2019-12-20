import csv
import os

import torch
from torch import Tensor
import numpy as np

from kge import Config, Configurable
from kge.indexing import create_default_index_functions
from kge.misc import kge_base_dir

from typing import Dict, List, Any, Callable, Union, Optional

# TODO add support to pickle dataset (and indexes) and reload from there
class Dataset(Configurable):
    """Stores information about a dataset.

    This includes the number of entities, number of relations, splits containing tripels
    (e.g., to train, validate, test), indexes, and various metadata about these objects.
    Most of these objects can be lazy-loaded on first use.

    """

    def __init__(self, config, folder=None):
        """Constructor for internal use.

        To load a dataset, use `Dataset.load()`."""
        super().__init__(config, "dataset")

        #: directory in which dataset is stored
        self.folder = folder

        # read the number of entities and relations from the config, if present
        try:
            self._num_entities: Int = config.get("dataset.num_entities")
        except KeyError:
            self._num_entities: Int = None

        try:
            self._num_relations: Int = config.get("dataset.num_relations")
        except KeyError:
            self._num_relations: Int = None

        #: split-name to (n,3) int32 tensor
        self._splits: Dict[str, Tensor] = {}

        #: meta data that is part if this dataset. Indexed by key.
        self._meta: Dict[str, Any] = {}

        #: data derived automatically from the splits or meta data. Indexed by key.
        self._indexes: Dict[str, Any] = {}

        #: functions that compute and add indexes as needed; arguments are dataset and
        # key. : Indexed by key (same key as in self._indexes)
        self.index_functions: Dict[str, Callable] = {}
        create_default_index_functions(self)

    ## LOADING ##########################################################################

    @staticmethod
    def load(config: Config, preload_data=True):
        """Loads a dataset.

        If preload_data is set, loads entity and relation maps as well as all splits.
        Otherwise, this data is lazy loaded on first use.

        """
        name = config.get("dataset.name")
        folder = os.path.join(kge_base_dir(), "data", name)
        if os.path.isfile(os.path.join(folder, "dataset.yaml")):
            config.log("Loading configuration of dataset " + name + "...")
            config.load(os.path.join(folder, "dataset.yaml"))

        dataset = Dataset(config, folder)
        if preload_data:
            dataset.entity_names()
            dataset.relation_names()
            for split in ["train", "valid", "test"]:
                dataset.split(split)
        return dataset

    @staticmethod
    def _load_map(
        filename: str, as_list: bool = False, delimiter: str = "\t"
    ) -> Union[List, Dict]:
        n = 0
        dictionary = {}
        with open(filename, "r") as file:
            for line in file:
                key, value = line.split(delimiter, maxsplit=1)
                if as_list:
                    key = int(key)
                    n = max(n, key + 1)
                dictionary[key] = value
        if as_list:
            array = [None] * n
            for index, value in dictionary.items():
                array[index] = value
            return array
        else:
            return dictionary

    @staticmethod
    def _load_triples(filename: str, delimiter="\t") -> Tensor:
        triples = np.loadtxt(filename, usecols=range(0, 3), dtype=int)
        return torch.from_numpy(triples)

    def shallow_copy(self):
        """Returns a dataset that shares the underlying splits and indexes.

        Changes to splits and indexes are also reflected on this and the copied dataset.
        """
        copy = Dataset(self.config, self.folder)
        copy._num_entities = self.num_entities()
        copy._num_relations = self.num_relations()
        copy._entities = self._entities
        copy._relations = self._relations
        copy._splits = self._splits
        copy._meta = self._meta
        copy._indexes = self._indexes
        return copy

    ## ACCESS ###########################################################################

    def split(self, split: str) -> Tensor:
        """Return the split of the specified name.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        if split not in self._splits:
            triples = Dataset._load_triples(
                os.path.join(self.folder, self.config.get(f"dataset.{split}"))
            )
            self._splits[split] = triples
            self.config.log(f"Loaded split {split} with {len(triples)} triples")

        return self._splits[split]

    def meta(self, key: str) -> Any:
        """Return metadata stored under the specified key."""
        return self._meta[key]

    def index(self, key: str) -> Any:
        """Return the index stored under the specified key.

        Index means any data structure that is derived from the dataset, including
        statistics and indexes.

        If the index has not yet been computed, computes it by calling the function
        specified in `self.index_functions`.

        """
        if key not in self._indexes:
            self.index_functions[key](self)
        return self._indexes[key]

    def train(self) -> Tensor:
        "Return training split."
        return self.split("train")

    def valid(self) -> Tensor:
        "Return validation split."
        return self.split("valid")

    def test(self) -> Tensor:
        "Return test split."
        return self.split("test")

    def entity_names(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode entity names.

        If `indexes` is `None`, return all names. If `indexes` is an integer, return the
        corresponding name. If `indexes` is a Tensor, return an ndarray of the same
        shape holding the corresponding names.

        """
        if "entities" not in self._meta:
            entities = Dataset._load_map(
                os.path.join(self.folder, self.config.get("dataset.entity_map")),
                as_list=True,
            )
            if self._num_entities and self._num_entities != len(entities):
                raise ValueError(
                    f"Expected {self._num_entities} entities, found {num_entities}"
                )
            self.config.log(f"Loaded map for {len(entities)} entities")
            self._meta["entities"] = entities

        return _decode_names(self._meta["entities"], indexes)

    def relation_names(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode relation names.

        If `indexes` is `None`, return all names. If `indexes` is an integer, return the
        corresponding name. If `indexes` is a Tensor, return an ndarray of the same
        shape holding the corresponding names.

        """
        if "relations" not in self._meta:
            relations = Dataset._load_map(
                os.path.join(self.folder, self.config.get("dataset.relation_map")),
                as_list=True,
            )
            if self._num_relations and self._num_relations != len(relations):
                raise ValueError(
                    f"Expected {self._num_relations} relations, found {num_relations}"
                )
            self.config.log(f"Loaded map for {len(relations)} relations")
            self._meta["relations"] = relations

        return _decode_names(self._meta["relations"], indexes)

    def num_entities(self) -> int:
        "Return the number of entities in this dataset."
        if not self._num_entities:
            self._num_entities = len(self.entity_names())
        return self._num_entities

    def num_relations(self) -> int:
        "Return the number of relations in this dataset."
        if not self._num_relations:
            self._num_relations = len(self.relation_names())
        return self._num_relations


def _decode_names(names, indexes):
    "Return the names corresponding to specified indexes"
    if indexes is None:
        return names
    elif isinstance(indexes, int):
        return names[indexes]
    else:
        shape = indexes.shape
        indexes = indexes.view(-1)
        names = np.array(list(map(lambda i: names[i], indexes)), dtype=str)
        return names.reshape(shape)
