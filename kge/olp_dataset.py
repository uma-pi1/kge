from __future__ import annotations

import os
import uuid

import torch
from torch import Tensor
import numpy as np
import pandas as pd
import pickle
import inspect
from kge.dataset import Dataset

from kge import Config, Configurable
import kge.indexing
from kge.indexing import create_default_index_functions
from kge.misc import kge_base_dir

from typing import Dict, List, Any, Callable, Union, Optional, Tuple


# TP: class to contain all information on an OLP dataset
class OLPDataset(Dataset):

    def __init__(self, config, folder=None):
        "Constructor for internal use. To load an OLP dataset, use OLPDataset.create()"

        super().__init__(config, folder)

        # read the number of tokens for entites from the config, if present
        try:
            self._num_tokens_entities: Int = config.get("dataset.num_tokens_entities")
            if self._num_tokens_entities < 0:
                self._num_tokens_entities = None
        except KeyError:
            self._num_tokens_entities: Int = None

        # read the number of tokens for relations from the config, if present
        try:
            self._num_tokens_relations: Int = config.get("dataset.num_tokens_relations")
            if self._num_tokens_relations < 0:
                self._num_tokens_relations = None
        except KeyError:
            self._num_tokens_relations: Int = None

        # read the maximum number of tokens per entity mention from the config, if present
        try:
            self._max_tokens_per_entity: Int = config.get("dataset.max_tokens_per_entity")
            if self._max_tokens_per_entity < 0:
                self._max_tokens_per_entity = None
        except KeyError:
            self._max_tokens_per_entity: Int = None

        # read the maximum number of tokens per relation mention from the config, if present
        try:
            self._max_tokens_per_relation: Int = config.get("dataset.max_tokens_per_relation")
            if self._max_tokens_per_relation < 0:
                self._max_tokens_per_relation = None
        except KeyError:
            self._max_tokens_per_relation: Int = None

        # Tensors containing the mappings of entity/relation ids to a series of token ids
        self._mentions_to_token_ids: Dict[str, Tensor] = {}

        # Tensors containing the alternative mentions for subjects and objects
        self._alternative_subject_mentions: Dict[str, Tensor] = {}
        self._alternative_object_mentions: Dict[str, Tensor] = {}

        #TODO: Check indexing and whether it comes up in the remaining pipeline. Necessary?

    # overwrite static method to create an OLPDataset
    @staticmethod
    def create(config: Config, preload_data: bool = True, folder: Optional[str] = None):
        name = config.get("dataset.name")
        if folder is None:
            folder = os.path.join(kge_base_dir(), "data", name)
        if os.path.isfile(os.path.join(folder, "dataset.yaml")):
            config.log("Loading configuration of dataset " + name + "...")
            config.load(os.path.join(folder, "dataset.yaml"))

        dataset = OLPDataset(config, folder)
        dataset2 = dataset.shallow_copy()
        if preload_data:
            # create mappings of ids to respective strings for mentions and tokens
            dataset.entity_ids()
            dataset.relation_ids()
            dataset.entity_token_ids()
            dataset.relation_token_ids()
            # create mappings of entity ids to a series of token ids
            dataset.entity_mentions_to_token_ids()
            dataset.relation_mentions_to_token_ids()

            for split in ["train", "valid", "test"]:
                dataset.split(split)

        return dataset

    # Return the number of tokens for entities in the OLP dataset
    def num_tokens_entities(self) -> int:
        "Return the number of tokens in the OLP dataset."
        if not self._num_tokens_entities:
            self._num_tokens_entities = len(self.entity_token_ids())
        return self._num_tokens_entities

    # Return the number of tokens for relations in the OLP dataset
    def num_tokens_relations(self) -> int:
        "Return the number of tokens in the OLP dataset."
        if not self._num_tokens_relations:
            self._num_tokens_relations = len(self.relation_token_ids())
        return self._num_tokens_relations

    # Return the max number of tokens per entity mention in the OLP dataset
    def max_tokens_per_entity(self) -> int:
        "Return the number of tokens in the OLP dataset."
        if not self._max_tokens_per_entity:
            self.entity_mentions_to_token_ids()
        return self._max_tokens_per_entity

    # Return the max number of tokens per entity mention in the OLP dataset
    def max_tokens_per_relation(self) -> int:
        "Return the number of tokens in the OLP dataset."
        if not self._max_tokens_per_relation:
            self.relation_mentions_to_token_ids()
        return self._max_tokens_per_relation

    # adjusted super method to get token id mappings for entities
    def entity_token_ids(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity ids.

        See `Dataset#map_indexes` for a description of the `indexes` argument.
        """
        return self.map_indexes(indexes, "entity_token_ids")

    # adjusted super method to get token id mappings for relations
    def relation_token_ids(
        self, indexes: Optional[Union[int, Tensor]] = None
    ) -> Union[str, List[str], np.ndarray]:
        """Decode indexes to entity ids.

        See `Dataset#map_indexes` for a description of the `indexes` argument.
        """
        return self.map_indexes(indexes, "relation_token_ids")

    # create mappings of entity mentions to a series of token ids
    def entity_mentions_to_token_ids(self):
        if "entities" not in self._alternative_object_mentions:
            map_, actual_max = self.load_token_sequences("entity_id_token_ids", self._num_entities, self._max_tokens_per_entity)
            self._mentions_to_token_ids["entities"] = torch.from_numpy(map_)
            self._max_tokens_per_entity = actual_max
        return self._mentions_to_token_ids["entities"]

    # create mappings of relation mentions to a series of token ids
    def relation_mentions_to_token_ids(self):
        if "relations" not in self._alternative_object_mentions:
            map_, actual_max = self.load_token_sequences("relation_id_token_ids", self._num_relations, self._max_tokens_per_relation)
            self._mentions_to_token_ids["relations"] = torch.from_numpy(map_)
            self._max_tokens_per_relation = actual_max
        return self._mentions_to_token_ids["relations"]

    def load_token_sequences(
        self,
        key: str,
        num_ids: int,
        max_tokens: int,
        id_delimiter: str = "\t",
        token_delimiter: str = " "
        # TODO: add pickle support
    ) -> Tuple[np.array, int]:
        """ Load a sequence of token ids associated with different mentions for a given key

        If duplicates are found, raise a key error as duplicates cannot be handled with the
        tensor structure of mention ids to token id sequences
        """

        self.ensure_available(key)
        filename = self.config.get(f"dataset.files.{key}.filename")
        filetype = self.config.get(f"dataset.files.{key}.type")

        if filetype != "sequence_map":
            raise TypeError(
                "Unexpected file type: "
                f"dataset.files.{key}.type='{filetype}', expected 'sequence_map'"
            )

        with open(os.path.join(self.folder, filename), "r") as file:
            dictionary = {}
            if num_ids and max_tokens:
                map_ = np.zeros([num_ids, max_tokens], dtype=int)
            actual_max = 0
            max_id = 0
            used_keys = set()
            for line in file:
                key, value = line.split(id_delimiter, maxsplit=1)
                value = value.rstrip("\n")
                try:
                    key = int(key)
                except ValueError:
                    raise TypeError(f"{filename} contains non-integer keys")
                if used_keys.__contains__(key):
                    raise KeyError(f"{filename} contains duplicated keys")
                used_keys.add(key)
                split_ = value.split(token_delimiter)
                actual_max = max(actual_max, len(split_))
                if num_ids and max_tokens:
                    map_[key][0:len(split_)] = split_
                else:
                    dictionary[key] = split_
                    max_id = max(max_id, key)

        if num_ids and max_tokens:
            map_ = np.delete(map_, np.s_[actual_max:map_.shape[1]], 1)
        else:
            map_ = np.zeros([max_id + 1, actual_max], dtype=int)
            for key, split_ in dictionary.items():
                map_[key][0:len(split_)] = split_

        self.config.log(f"Loaded {map_.shape[0]} token sequences from {key}")

        return map_, actual_max

    def split(self, split: str) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the split and the alternative mentions of the specified name.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples and two NxA IntTensors (whereas A is the maximum number of
        alternative mentions).

        """
        return self.load_quintuples(split)

    def load_quintuples(self, key: str) -> Tensor:
        "Load or return the triples and alternative mentions with the specified key."
        if key not in self._triples:
            self.ensure_available(key)
            filename = self.config.get(f"dataset.files.{key}.filename")
            filetype = self.config.get(f"dataset.files.{key}.type")
            if filetype != "triples" and filetype != "quintuples":
                raise ValueError(
                    "Unexpected file type: "
                    f"dataset.files.{key}.type='{filetype}', expected 'triples' or 'quintuples'"
                )
            triples, alternative_subjects, alternative_objects = OLPDataset._load_quintuples(
                os.path.join(self.folder, filename),
                filetype,
                use_pickle=self.config.get("dataset.pickle"),
            )
            self.config.log(f"Loaded {len(triples)} {key} {filetype}")
            self._triples[key] = triples
            self._alternative_subject_mentions[key] = alternative_subjects
            self._alternative_object_mentions[key] = alternative_objects

        return self._triples[key], self._alternative_subject_mentions[key], self._alternative_object_mentions[key]

    @staticmethod
    def _load_quintuples(
        filename: str,
        filetype: str,
        col_delimiter="\t",
        id_delimiter=" ",
        use_pickle=False) -> Tuple[Tensor, Tensor, Tensor]:
        #TODO: add pickle support

        """
        Read the tuples and alternative mentions from the specified file.

        If filetype is triples (no alternative mentions available), save the correct answer
        within the alternative mentions tensor.
        """

        # numpy loadtxt is very slow, use pandas instead
        data = pd.read_csv(
            filename, sep=col_delimiter, header=None, usecols=range(0, 5)
        )
        triples = data.iloc[:, [0, 1, 2]].to_numpy()
        if filetype == "triples":
            alternative_subject_mentions = triples[:, 0]
            alternative_object_mentions = triples[:, 2]
        else:
            subject_mentions = [None] * len(data.index)
            object_mentions = [None] * len(data.index)
            max_subject_mentions = 0
            max_object_mentions = 0
            for i, (subject, object) in enumerate(data.iloc[:, [3, 4]].values.tolist()):
                subject_split = subject.split(id_delimiter)
                subject_mentions[i] = subject_split
                max_subject_mentions = max(max_subject_mentions, len(subject_split))

                object_split = object.split(id_delimiter)
                object_mentions[i] = object_split
                max_object_mentions = max(max_object_mentions, len(object_split))

            alternative_subject_mentions = np.zeros([triples.shape[0], max_subject_mentions], dtype=int)
            alternative_object_mentions = np.zeros([triples.shape[0], max_object_mentions], dtype=int)
            for i, (subject_mention, object_mention) in enumerate(zip(subject_mentions, object_mentions)):
                alternative_subject_mentions[i][0:len(subject_mention)] = subject_mention
                alternative_object_mentions[i][0:len(object_mention)] = object_mention

        return torch.from_numpy(triples), torch.from_numpy(alternative_subject_mentions), torch.from_numpy(alternative_object_mentions)

    # adjusted super method to also copy new OLPDataset variables
    def shallow_copy(self):
        """Returns a dataset that shares the underlying splits and indexes.

        Changes to splits and indexes are also reflected on this and the copied dataset.
        """
        copy = OLPDataset(self.config, self.folder)
        copy._num_entities = self.num_entities()
        copy._num_relations = self.num_relations()
        copy._num_tokens_entities = self.num_tokens_entities()
        copy._num_tokens_relations = self.num_tokens_relations()
        copy._max_tokens_per_entity = self.max_tokens_per_entity()
        copy._max_tokens_per_relation = self.max_tokens_per_relation()
        copy._triples = self._triples
        copy._mentions_to_token_ids = self._mentions_to_token_ids
        copy._alternative_subject_mentions = self._alternative_subject_mentions
        copy._alternative_object_mentions = self._alternative_object_mentions
        copy._meta = self._meta
        copy._indexes = self._indexes
        copy.index_functions = self.index_functions
        return copy

    # TODO: methods that have not been adjusted (as not necessary atm or will be understood better later on):
    # - create from and save_to (loads/saves a dataset from/to a checkpoint)
    # - pickle-related methods
    # - indexing functions
