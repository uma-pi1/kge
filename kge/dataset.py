import csv
import os
import sys

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
            if self._num_entities < 0:
                self._num_entities = None
        except KeyError:
            self._num_entities: Int = None

        try:
            self._num_relations: Int = config.get("dataset.num_relations")
            if self._num_relations < 0:
                self._num_relations = None
        except KeyError:
            self._num_relations: Int = None

        #: split-name to (n,3) int32 tensor
        self._triples: Dict[str, Tensor] = {}

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
            dataset.entity_ids()
            dataset.relation_ids()
            for split in ["train", "valid", "test"]:
                dataset.split(split)
        return dataset

    @staticmethod
    def _load_triples(filename: str, delimiter="\t") -> Tensor:
        triples = np.loadtxt(filename, usecols=range(0, 3), dtype=int)
        return torch.from_numpy(triples)

    def load_triples(self, key: str) -> Tensor:
        "Load or return the triples with the specified key."
        if key not in self._triples:
            filename = self.config.get(f"dataset.files.{key}.filename")
            filetype = self.config.get(f"dataset.files.{key}.type")
            if filetype != "triples":
                raise ValueError(
                    "Unexpected file type: "
                    f"dataset.files.{key}.type='{filetype}', expected 'triples'"
                )
            triples = Dataset._load_triples(os.path.join(self.folder, filename))
            self.config.log(f"Loaded {len(triples)} {key} triples")
            self._triples[key] = triples

        return self._triples[key]

    @staticmethod
    def _load_map(
        filename: str,
        as_list: bool = False,
        delimiter: str = "\t",
        ignore_duplicates=False,
    ) -> Union[List, Dict]:
        n = 0
        dictionary = {}
        warned_overrides = False
        duplicates = 0
        with open(filename, "r") as file:
            for line in file:
                key, value = line.split(delimiter, maxsplit=1)
                value = value.rstrip("\n")
                if as_list:
                    key = int(key)
                    n = max(n, key + 1)
                if key in dictionary:
                    duplicates += 1
                    if not ignore_duplicates:
                        raise KeyError(f"{filename} contains duplicated keys")
                else:
                    dictionary[key] = value
        if as_list:
            array = [None] * n
            for index, value in dictionary.items():
                array[index] = value
            return array, duplicates
        else:
            return dictionary, duplicates

    def load_map(
        self,
        key: str,
        as_list: bool = False,
        maptype=None,
        ids_key=None,
        ignore_duplicates=False,
    ) -> Union[List, Dict]:
        """Load or return the map with the specified key.

        If `as_list` is set, the map is converted to an array indexed by the map's keys.

        If `maptype` is set ensures that the map being loaded has the specified type.
        Valid map types are `map` (keys are indexes) and `idmap` (keys are ids).

        If the map is of type `idmap`, its keys can be converted to indexes by setting
        `ids_key` to either `entity_ids` or `relation_ids` and `as_list` to `True`.

        If ignore_duplicates is set to `False` and the map contains duplicate keys,
        raise a `KeyError`. Otherwise, logs a warning and picks first occurrence of a
        key.

        """
        if key not in self._meta:
            filename = self.config.get(f"dataset.files.{key}.filename")
            filetype = self.config.get(f"dataset.files.{key}.type")
            if (maptype and filetype != maptype) or (
                not maptype and filetype not in ["map", "idmap"]
            ):
                if not maptype:
                    maptype = "map' or 'idmap"
                raise ValueError(
                    "Unexpected file type: "
                    f"dataset.files.{key}.type='{filetype}', expected {maptype}"
                )
            if filetype == "idmap" and as_list and ids_key:
                map_, duplicates = Dataset._load_map(
                    os.path.join(self.folder, filename),
                    as_list=False,
                    ignore_duplicates=ignore_duplicates,
                )
                ids = self.load_map(ids_key, as_list=True)
                map_ = [map_.get(ids[i], None) for i in range(len(ids))]
                nones = map_.count(None)
                if nones > 0:
                    self.config.log(
                        f"Warning: could not find {nones} ids in map {key}; "
                        "filling with None."
                    )
            else:
                map_, duplicates = Dataset._load_map(
                    os.path.join(self.folder, filename),
                    as_list=as_list,
                    ignore_duplicates=ignore_duplicates,
                )

            if duplicates > 0:
                self.config.log(
                    f"Warning: map {key} contains {duplicates} duplicate keys, "
                    "all which have been ignored"
                )
            self.config.log(f"Loaded {len(map_)} keys from map {key}")
            self._meta[key] = map_

        return self._meta[key]

    def shallow_copy(self):
        """Returns a dataset that shares the underlying splits and indexes.

        Changes to splits and indexes are also reflected on this and the copied dataset.
        """
        copy = Dataset(self.config, self.folder)
        copy._num_entities = self.num_entities()
        copy._num_relations = self.num_relations()
        copy._triples = self._triples
        copy._meta = self._meta
        copy._indexes = self._indexes
        copy.index_functions = self.index_functions
        return copy

    ## ACCESS ###########################################################################

    def num_entities(self) -> int:
        "Return the number of entities in this dataset."
        if not self._num_entities:
            self._num_entities = len(self.entity_ids())
        return self._num_entities

    def num_relations(self) -> int:
        "Return the number of relations in this dataset."
        if not self._num_relations:
            self._num_relations = len(self.relation_ids())
        return self._num_relations

    def split(self, split: str) -> Tensor:
        """Return the split of the specified name.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        return self.load_triples(split)

    def train(self) -> Tensor:
        """Return training split.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        return self.split("train")

    def valid(self) -> Tensor:
        """Return validation split.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        return self.split("valid")

    def test(self) -> Tensor:
        """Return test split.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        return self.split("test")

    def entity_ids(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity ids.

        See `Dataset#map_indexes` for a description of the `indexes` argument.
        """
        return self.map_indexes(indexes, "entity_ids")

    def relation_ids(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to relation ids.

        See `Dataset#map_indexes` for a description of the `indexes` argument.
        """
        return self.map_indexes(indexes, "relation_ids")

    def entity_strings(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity strings.

        See `Dataset#map_indexes` for a description of the `indexes` argument.

        """
        map_ = self.load_map(
            "entity_strings", as_list=True, ids_key="entity_ids", ignore_duplicates=True
        )
        return self._map_indexes(indexes, map_)

    def relation_strings(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to relation strings.

        See `Dataset#map_indexes` for a description of the `indexes` argument.

        """
        map_ = self.load_map(
            "relation_strings",
            as_list=True,
            ids_key="relation_ids",
            ignore_duplicates=True,
        )
        return self._map_indexes(indexes, map_)

    def meta(self, key: str) -> Any:
        """Return metadata stored under the specified key."""
        return self._meta[key]

    def index(self, key: str) -> Any:
        """Return the index stored under the specified key.

        Index means any data structure that is derived from the dataset, including
        statistics and indexes.

        If the index has not yet been computed, computes it by calling the function
        specified in `self.index_functions`.

        See `kge.indexing.create_default_index_functions()` for the indexes available by
        default.

        """
        if key not in self._indexes:
            self.index_functions[key](self)
        return self._indexes[key]

    @staticmethod
    def _map_indexes(indexes, values):
        "Return the names corresponding to specified indexes"
        if indexes is None:
            return values
        elif isinstance(indexes, int):
            return values[indexes]
        else:
            shape = indexes.shape
            indexes = indexes.view(-1)
            names = np.array(list(map(lambda i: values[i], indexes)), dtype=str)
            return names.reshape(shape)

    def map_indexes(
        self, indexes: Optional[Union[int, Tensor]], key: str
    ) -> Union[Any, List[Any], np.ndarray]:
        """Maps indexes to values using the specified map.

        `key` refers to the key of a map file of the dataset, which associates a value
        with each numerical index. The map file is loaded automatically.

        If `indexes` is `None`, return all values. If `indexes` is an integer, return
        the corresponding value. If `indexes` is a Tensor, return an ndarray of the same
        shape holding the corresponding values.

        """
        map_ = self.load_map(key, as_list=True)
        return Dataset._map_indexes(indexes, map_)
