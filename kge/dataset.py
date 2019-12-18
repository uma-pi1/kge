import csv
import os
from collections import defaultdict, OrderedDict

import torch
from torch import Tensor
import numpy as np

from kge import Config, Configurable
from kge.misc import kge_base_dir

from typing import Dict, List

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

        #: # entity index -> metadata array of strings
        self._entities: List[List[str]] = None

        #: # relations index -> metadata array of strings
        self._relations: List[List[str]] = None

        #: split-name to (n,3) int32 tensor
        self._splits: Dict[str, Tensor] = {}

        #: split-name to array, which maps triple index -> metadata array of strings
        self._splits_meta: Dict[str, List[List[str]]] = {}

        self.indexes = {}  # map: name of index -> index (used mainly by training jobs)

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
            dataset.entities()
            dataset.relations()
            for split in ["train", "valid", "test"]:
                dataset.split(split)
        return dataset

    @staticmethod
    def _load_map(filename):
        n = 0
        dictionary = {}
        with open(filename, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                index = int(row[0])
                meta = row[1:]
                dictionary[index] = meta
                n = max(n, index + 1)
        array = [[]] * n
        for index, meta in dictionary.items():
            array[index] = meta
        return n, array

    @staticmethod
    def _load_triples(filename):
        n = 0
        triples = np.loadtxt(filename, delimiter="\t", usecols=range(0, 3), dtype=int)
        triples = torch.from_numpy(triples)
        num_lines = triples.shape[0]
        meta = [[]] * num_lines
        with open(filename, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                meta[n] = row[3:]
                n += 1

        return triples, meta

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
        copy._splits_meta = self._splits_meta
        copy.indexes = self.indexes
        return copy

    ## ACCESS ###########################################################################

    def split(self, split: str) -> Tensor:
        """Return the split of the specified name.

        If the split is not yet loaded, load it. Returns an Nx3 IntTensor of
        spo-triples.

        """
        if split not in self._splits:
            triples, triples_meta = Dataset._load_triples(
                os.path.join(self.folder, self.config.get(f"dataset.{split}"))
            )
            self._splits[split] = triples
            self._splits_meta[split] = triples_meta
            self.config.log(f"Loaded split {split} with {len(triples)} triples")

        return self._splits[split]

    def train(self) -> Tensor:
        "Return training split."
        return self.split("train")

    def valid(self) -> Tensor:
        "Return validation split."
        return self.split("valid")

    def test(self) -> Tensor:
        "Return test split."
        return self.split("test")

    def entities(self) -> List[List[str]]:
        "Return an array that holds for each entity a list of metadata strings"
        if not self._entities:
            num_entities, entities = Dataset._load_map(
                os.path.join(self.folder, self.config.get("dataset.entity_map"))
            )
            if self._num_entities and self._num_entities != num_entities:
                raise ValueError(
                    f"Expected {self._num_entities} entities, found {num_entities}"
                )
            self.config.log(f"Loaded map for {num_entities} entities")
            self._entities = entities

        return self._entities

    def relations(self) -> List[List[str]]:
        "Return an array that holds for each entity a list of metadata strings"
        if not self._relations:
            num_relations, relations = Dataset._load_map(
                os.path.join(self.folder, self.config.get("dataset.relation_map"))
            )
            if self._num_relations and self._num_relations != num_relations:
                raise ValueError(
                    f"Expected {self._num_relations} relations, found {num_relations}"
                )
            self.config.log(f"Loaded map for {num_relations} relations")
            self._relations = relations

        return self._relations

    def num_entities(self) -> int:
        "Return the number of entities in this dataset."
        if not self._num_entities:
            self._num_entities = len(self.entities())
        return self._num_entities

    def num_relations(self) -> int:
        "Return the number of relations in this dataset."
        if not self._num_relations:
            self._num_relations = len(self.relations())
        return self._num_relations

    ## INDEXING #########################################################################

    def index_KvsAll(self, split: str, sp_po: str):
        """Return an index for the triples in split (''train'', ''valid'', ''test'')
        from the specified constituents (''sp'' or ''po'') to the indexes of the
        remaining constituent (''o'' or ''s'', respectively.)

        The index maps from `tuple' to `torch.LongTensor`.

        The index is cached in the provided dataset under name ''split_sp_po''. If
        this index is already present, does not recompute it.

        """
        triples = self.split(split)
        if sp_po == "sp":
            sp_po_cols = [0, 1]
            value_column = 2
        elif sp_po == "po":
            sp_po_cols = [1, 2]
            value_column = 0
        else:
            raise ValueError()

        name = split + "_" + sp_po
        if not self.indexes.get(name):
            self.indexes[name] = Dataset.group_by_sp_po(
                triples[:, sp_po_cols], triples[:, value_column]
            )
            self.config.log(
                "{} distinct {} pairs in {}".format(
                    len(self.indexes[name]), sp_po, split
                ),
                prefix="  ",
            )

        return self.indexes.get(name)

    @staticmethod
    def group_by_sp_po(sp_po_list, o_s_list) -> dict:
        result = defaultdict(list)
        for sp_po, o_s in zip(sp_po_list.tolist(), o_s_list.tolist()):
            result[tuple(sp_po)].append(o_s)
        for sp_po, o_s in result.items():
            result[sp_po] = torch.IntTensor(sorted(o_s))
        return OrderedDict(result)

    @staticmethod
    def prepare_index(index):
        sp_po = torch.tensor(list(index.keys()), dtype=torch.int)
        o_s = torch.cat(list(index.values()))
        offsets = torch.cumsum(
            torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
        )
        return sp_po, o_s, offsets

    def index_relation_types(self):
        """
        create dictionary mapping from {1-N, M-1, 1-1, M-N} -> set of relations
        """
        if "relation_types" in self.indexes:
            return
        relation_types = self._get_relation_types()
        relations_per_type = {}
        for k, v in relation_types.items():
            relations_per_type.setdefault(v, set()).add(k)
        for k, v in relations_per_type.items():
            self.config.log("{} relations of type {}".format(len(v), k), prefix="  ")
        self.indexes["relation_types"] = relation_types
        self.indexes["relations_per_type"] = relations_per_type

    def _get_relation_types(self,):
        """
        Classify relations into 1-N, M-1, 1-1, M-N

        Bordes, Antoine, et al.
        "Translating embeddings for modeling multi-relational data."
        Advances in neural information processing systems. 2013.

        :return: dictionary mapping from int -> {1-N, M-1, 1-1, M-N}
        """
        relation_stats = torch.zeros((self.num_relations(), 6))
        for index, p in [
            (self.index_KvsAll("train", "sp"), 1),
            (self.index_KvsAll("train", "po"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = labels.float().sum()
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        result = dict()
        for i, relation in enumerate(self.relations()):
            result[i] = "{}-{}".format(
                "1" if relation_stats[i, 4].item() == 0 else "M",
                "1" if relation_stats[i, 5].item() == 0 else "N",
            )
        return result

    # TODO this is metadata; refine API
    def index_frequency_percentiles(self, recompute=False):
        """
        :return: dictionary mapping from
        {
         'subject':
            {25%, 50%, 75%, top} -> set of entities
         'relations':
            {25%, 50%, 75%, top} -> set of relations
         'object':
            {25%, 50%, 75%, top} -> set of entities
        }
        """
        if "frequency_percentiles" in self.indexes:
            return
        subject_stats = torch.zeros((self.num_entities(), 1))
        relation_stats = torch.zeros((self.num_relations(), 1))
        object_stats = torch.zeros((self.num_entities(), 1))
        for (s, p, o) in self.train():
            subject_stats[s] += 1
            relation_stats[p] += 1
            object_stats[o] += 1
        result = dict()
        for arg, stats, num in [
            (
                "subject",
                [
                    i
                    for i, j in list(
                        sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                    )
                ],
                self.num_entities(),
            ),
            (
                "relation",
                [
                    i
                    for i, j in list(
                        sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                    )
                ],
                self.num_relations(),
            ),
            (
                "object",
                [
                    i
                    for i, j in list(
                        sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                    )
                ],
                self.num_entities(),
            ),
        ]:
            for percentile, (begin, end) in [
                ("25%", (0.0, 0.25)),
                ("50%", (0.25, 0.5)),
                ("75%", (0.5, 0.75)),
                ("top", (0.75, 1.0)),
            ]:
                if arg not in result:
                    result[arg] = dict()
                result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
        self.indexes["frequency_percentiles"] = result
