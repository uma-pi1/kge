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
import time

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
        self._alternative_subject_mentions: Dict[str, List] = {}
        self._alternative_object_mentions: Dict[str, List] = {}
        self._nr_alternative_subjects: Dict[str, Int] = {}
        self._nr_alternative_objects: Dict[str, Int] = {}

        # Dictionary that maps triples to their index
        self._triple_indexes: Dict[str, Dict] = {}

        # TODO: Check indexing and how it comes up in the remaining pipeline. Create new indizes as necessary.
        self.index_functions: Dict[str, Callable] = {}
        create_default_index_functions(self)

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
                dataset.split_olp(split)

        return dataset

    def vocab_size_entities(self) -> int:
        "Return the number of embeddings for entities given the dataset."
        return self.num_tokens_entities()

    def vocab_size_relations(self) -> int:
        "Return the number of embeddings for relations given the dataset."
        return self.num_tokens_relations()

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

    # Return the number of alternative subject mentions for a key
    def nr_alternative_subjects(self, key: str) -> int:
        "Return the number of alternative subjects for the given key in the OLP dataset."
        if not key in self._nr_alternative_subjects.keys() or self._nr_alternative_subjects[key] is None:
            self._nr_alternative_subjects[key] = sum(
                1 if tensor.ndim == 1 else tensor.shape[0] for tensor in self._alternative_subject_mentions[key])
        return self._nr_alternative_subjects[key]

        # Return the number of alternative object mentions for a key

    def nr_alternative_objects(self, key: str) -> int:
        "Return the number of alternative objects for the given key in the OLP dataset."
        if not key in self._nr_alternative_objects.keys() or self._nr_alternative_objects[key] is None:
            self._nr_alternative_objects[key] = sum(
                1 if tensor.ndim == 1 else tensor.shape[0] for tensor in self._alternative_object_mentions[key])
        return self._nr_alternative_objects[key]

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
            map_, actual_max = self.load_token_sequences("entity_id_token_ids", self._num_entities,
                                                         self._max_tokens_per_entity)
            self._mentions_to_token_ids["entities"] = torch.from_numpy(map_)
            self._max_tokens_per_entity = actual_max
        return self._mentions_to_token_ids["entities"]

    # create mappings of relation mentions to a series of token ids_nr_alternative_subjects
    def relation_mentions_to_token_ids(self):
        if "relations" not in self._alternative_object_mentions:
            map_, actual_max = self.load_token_sequences("relation_id_token_ids", self._num_relations,
                                                         self._max_tokens_per_relation)
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
                k, value = line.split(id_delimiter, maxsplit=1)
                value = value.rstrip("\n")
                try:
                    k = int(k)
                except ValueError:
                    raise TypeError(f"{filename} contains non-integer keys")
                if used_keys.__contains__(k):
                    raise KeyError(f"{filename} contains duplicated keys")
                used_keys.add(k)
                split_ = [int(i) for i in value.split(token_delimiter)]
                if self.config.get("dataset.filter_start_and_end_token"):
                    split_ = split_[1:len(split_) - 1]
                actual_max = max(actual_max, len(split_))
                if num_ids and max_tokens:
                    map_[k][0:len(split_)] = split_
                else:
                    dictionary[k] = split_
                    max_id = max(max_id, k)

        if num_ids and max_tokens:
            map_ = np.delete(map_, np.s_[actual_max:map_.shape[1]], 1)
        else:
            map_ = np.zeros([max_id + 1, actual_max], dtype=int)
            for k, split_ in dictionary.items():
                map_[k][0:len(split_)] = split_

        self.config.log(f"Loaded {map_.shape[0]} token sequences from {key}")

        return map_, actual_max

    def split_olp(self, split: str) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the split and the alternative mentions of the specified name.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples and two NxA IntTensors (whereas A is the maximum number of
        alternative mentions).

        """
        return self.load_quintuples(split)

    def load_quintuples(self, key: str) -> Tuple[Tensor, List, List]:
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
            triples, triple_indexes, alternative_subjects, alternative_objects, num_subjects, num_objects = OLPDataset._load_quintuples(
                os.path.join(self.folder, filename),
                filetype,
                use_pickle=self.config.get("dataset.pickle"),
            )
            self.config.log(f"Loaded {len(triples)} {key} {filetype}")
            self._triples[key] = triples
            self._triple_indexes[key] = triple_indexes
            self._alternative_subject_mentions[key] = alternative_subjects
            self._alternative_object_mentions[key] = alternative_objects
            self._nr_alternative_subjects[key] = num_subjects
            self._nr_alternative_objects[key] = num_objects

        return self._triples[key], self._alternative_subject_mentions[key], self._alternative_object_mentions[key]

    @staticmethod
    def _load_quintuples(
            filename: str,
            filetype: str,
            col_delimiter="\t",
            id_delimiter=" ",
            use_pickle=False) -> Tuple[Tensor, Dict, List, List, Int, Int]:

        """
        Read the tuples and alternative mentions from the specified file.

        If filetype is triples (no alternative mentions available), save the correct answer
        within the alternative mentions tensor.
        """
        if use_pickle:
            # check if there is a pickled, up-to-date version of the file
            pickle_suffix = Dataset._to_valid_filename(f"-{col_delimiter}.pckl")
            pickle_filename = filename + pickle_suffix
            pickle_filename_triple_indizes = pickle_filename.replace(".pckl", "-ti.pckl")
            alternative_subject_mention_pickle_filename = pickle_filename.replace(".pckl", "-asm.pckl")
            alternative_object_mention_pickle_filename = pickle_filename.replace(".pckl", "-aom.pckl")
            triples = Dataset._pickle_load_if_uptodate(None, pickle_filename, filename)
            triple_indexes = Dataset._pickle_load_if_uptodate(None, pickle_filename_triple_indizes, filename)
            alternative_subject_mentions = Dataset._pickle_load_if_uptodate(None,
                                                                            alternative_subject_mention_pickle_filename,
                                                                            filename)
            alternative_object_mentions = Dataset._pickle_load_if_uptodate(None,
                                                                           alternative_object_mention_pickle_filename,
                                                                           filename)
            if triples is not None and triple_indexes is not None and alternative_subject_mentions is not None and alternative_object_mentions is not None:
                return triples, triple_indexes, alternative_subject_mentions, alternative_object_mentions, None, None
                # numpy loadtxt is very slow, use pandas instead
        data = pd.read_csv(
            filename, sep=col_delimiter, header=None, usecols=range(0, 5)
        )
        triple_indexes: Dict[tuple, int] = {}
        if filetype == "triples":
            triples = torch.tensor(data.loc[:, 0:2].values)
            alternative_subject_mentions = []#list(torch.split(torch.cat([triples, triples[:, 0].view(-1, 1)], dim=1), 1))
            alternative_object_mentions = []#list(torch.split(torch.cat([triples, triples[:, 2].view(-1, 1)], dim=1), 1))
            sum_subject_mentions = triples.shape[0]
            sum_object_mentions = triples.shape[0]
        else:
            triples = np.empty((data.shape[0], 3), int)
            alternative_subject_mentions = [None] * data.shape[0]
            alternative_object_mentions = [None] * data.shape[0]
            sum_subject_mentions = 0
            sum_object_mentions = 0
            i = 0
            for (sub, pred, obj, alt_subject, alt_object) in zip(data[0], data[1], data[2], data[3], data[4]):
                alt_subjects = [int(i) for i in alt_subject.split(id_delimiter) if not int(i) < 0]
                alt_objects = [int(i) for i in alt_object.split(id_delimiter) if not int(i) < 0]
                entry = (sub, pred, obj)
                triples[i] = entry
                triple_indexes[entry] = i

                # build data structure for alternative mentions of Tensor (n * 4)
                # n = nr triples, columns: subject, predicate, object, alternative mentions
                if len(alt_subjects) == 1:
                    alternative_subject_mentions[i] = torch.tensor([sub, pred, obj, *alt_subjects], dtype=int).view(-1,
                                                                                                                    4).cpu()
                else:
                    alternative_subject_mentions[i] = torch.cat(
                        [torch.as_tensor(entry, dtype=int).repeat((len(alt_subjects), 1)),
                         torch.as_tensor(alt_subjects).view(-1, 1)], dim=1)
                sum_subject_mentions += len(alt_subjects)
                if len(alt_objects) == 1:
                    alternative_object_mentions[i] = torch.tensor([sub, pred, obj, *alt_objects], dtype=int).view(-1, 4).cpu()
                else:
                    alternative_object_mentions[i] = torch.cat(
                        [torch.as_tensor(entry, dtype=int).repeat((len(alt_objects), 1)),
                         torch.as_tensor(alt_objects).view(-1, 1)], dim=1)
                sum_object_mentions += len(alt_objects)
                i += 1
            triples = torch.from_numpy(triples)

        if use_pickle:
            Dataset._pickle_dump_atomic(triples, pickle_filename)
            Dataset._pickle_dump_atomic(triple_indexes, pickle_filename_triple_indizes)
            Dataset._pickle_dump_atomic(alternative_subject_mentions, alternative_subject_mention_pickle_filename)
            Dataset._pickle_dump_atomic(alternative_object_mentions, alternative_object_mention_pickle_filename)

        return triples, triple_indexes, alternative_subject_mentions, alternative_object_mentions, sum_subject_mentions, sum_object_mentions

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
