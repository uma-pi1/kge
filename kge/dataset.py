import csv
import os
from collections import defaultdict, OrderedDict

import torch

from kge.util.misc import kge_base_dir


# TODO add support to pickle dataset (and indexes) and reload from there
class Dataset:
    def __init__(
        self,
        config,
        num_entities,
        entities,
        num_relations,
        relations,
        train,
        train_meta,
        valid,
        valid_meta,
        test,
        test_meta,
    ):
        self.config = config
        self.num_entities = num_entities
        self.entities = entities  # array: entity index -> metadata array of strings
        self.num_relations = num_relations
        self.relations = relations  # array: relation index -> metadata array of strings
        self.train = train  # (n,3) int32 tensor
        self.train_meta = (
            train_meta
        )  # array: triple row number -> metadata array of strings
        self.valid = valid  # (n,3) int32 tensor
        self.valid_meta = (
            valid_meta
        )  # array: triple row number -> metadata array of strings
        self.test = test  # (n,3) int32 tensor
        self.test_meta = (
            test_meta
        )  # array: triple row number -> metadata array of strings
        self.indexes = {}  # map: name of index -> index (used mainly by training jobs)

    @staticmethod
    def load(config):
        name = config.get("dataset.name")
        config.log("Loading dataset " + name + "...")
        base_dir = os.path.join(kge_base_dir(), "data/" + name)

        num_entities, entities = Dataset._load_map(
            os.path.join(base_dir, config.get("dataset.entity_map"))
        )
        config.log(str(num_entities) + " entities", prefix="  ")
        num_relations, relations = Dataset._load_map(
            os.path.join(base_dir, config.get("dataset.relation_map"))
        )
        config.log(str(num_relations) + " relations", prefix="  ")

        train, train_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.train"))
        )
        config.log(str(len(train)) + " training triples", prefix="  ")

        valid, valid_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.valid"))
        )
        config.log(str(len(valid)) + " validation triples", prefix="  ")

        test, test_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.test"))
        )
        config.log(str(len(test)) + " test triples", prefix="  ")

        return Dataset(
            config,
            num_entities,
            entities,
            num_relations,
            relations,
            train,
            train_meta,
            valid,
            valid_meta,
            test,
            test_meta,
        )

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
        dictionary = {}
        with open(filename, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                s = int(row[0])
                p = int(row[1])
                o = int(row[2])
                meta = row[3:]
                dictionary[n] = (torch.IntTensor([s, p, o]), meta)
                n += 1
        triples = torch.empty(n, 3, dtype=torch.int32)
        meta = [[]] * n
        for index, value in dictionary.items():
            triples[index, :] = value[0]
            meta[index] = value[1]
        return triples, meta

    def index_1toN(self, split: str, sp_po: str):
        """Return an index for the triples in split (''train'', ''valid'', ''test'')
        from the specified constituents (''sp'' or ''po'') to the indexes of the
        remaining constituent (''o'' or ''s'', respectively.)

        The index maps from `tuple' to `torch.LongTensor`.

        The index is cached in the provided dataset under name ''split_sp_po''. If
        this index is already present, does not recompute it.

        """
        if split == "train":
            triples = self.train
        elif split == "valid":
            triples = self.valid
        elif split == "test":
            triples = self.test
        else:
            raise ValueError()

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
                "{} distinct {} pairs in {}".format(len(self.indexes[name]), sp_po, split), prefix="  "
            )

        return self.indexes.get(name)

    @staticmethod
    def group_by_sp_po(sp_po_list, o_s_list) -> dict:
        result = defaultdict(list)
        for sp_po,o_s in zip(sp_po_list.tolist(), o_s_list.tolist()):
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