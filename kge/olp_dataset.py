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

from typing import Dict, List, Any, Callable, Union, Optional


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

        #: Tensors containing the mappings of entity/relation ids to a series of token ids
        self._mentions_to_token_ids: Dict[str, Tensor] = {}

        #TODO: Understand / Check if necessary
        #: functions that compute and add indexes as needed; arguments are dataset and
        #: key. Index functions are expected to not recompute an index that is already
        #: present. Indexed by key (same key as in self._indexes)
        # self.index_functions: Dict[str, Callable] = {}
        # create_default_index_functions(self)

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

            #TODO: add alternative mentions for validation and test
            for split in ["train", "valid", "test"]:
                dataset.split(split)

        return dataset


    # TP: Return the number of tokens in the OLP dataset
    def num_tokens(self) -> int:
        "Return the number of tokens in the OLP dataset."
        if not self._num_tokens:
            self._num_tokens = 0  # TODO: finish function
        return self._num_tokens

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
        self._mentions_to_token_ids["entities"] = torch.zeros([self._num_entities, self._max_tokens_per_entity])

    # create mappings of relation mentions to a series of token ids
    def relation_mentions_to_token_ids(self):
        self._mentions_to_token_ids["relations"] = torch.zeros([self._num_relations, self._max_tokens_per_entity])

