import csv
import os
from collections import defaultdict, OrderedDict

import torch
import numpy as np

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
        num_relations, relations = Dataset._load_map(
            os.path.join(base_dir, config.get("dataset.relation_map"))
        )

        train, train_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.train"))
        )

        valid, valid_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.valid"))
        )

        test, test_meta = Dataset._load_triples(
            os.path.join(base_dir, config.get("dataset.test"))
        )

        result = Dataset(
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

        config.log(str(num_entities) + " entities", prefix="  ")
        config.log(str(num_relations) + " relations", prefix="  ")
        config.log(str(len(train)) + " training triples", prefix="  ")
        config.log(str(len(valid)) + " validation triples", prefix="  ")
        config.log(str(len(test)) + " test triples", prefix="  ")

        return result

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

    def index_KvsAll(self, split: str, sp_po: str):
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
        for k,v in relations_per_type.items():
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
        relation_stats = torch.zeros((self.num_relations, 6))
        for index, p in [
            (self.index_KvsAll('train', 'sp'), 1),
            (self.index_KvsAll('train', 'po'), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0+p*2] = labels.float().sum()
                relation_stats[prefix[p], 1+p*2] = relation_stats[prefix[p], 1+p*2] + 1.
        relation_stats[:,4] = (relation_stats[:,0]/relation_stats[:,1]) > 1.5
        relation_stats[:,5] = (relation_stats[:,2]/relation_stats[:,3]) > 1.5
        result = dict()
        for i, relation in enumerate(self.relations):
            result[i] = '{}-{}'.format(
                '1' if relation_stats[i,4].item() == 0 else 'M',
                '1' if relation_stats[i,5].item() == 0 else 'N', )
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
        subject_stats = torch.zeros((self.num_entities, 1))
        relation_stats = torch.zeros((self.num_relations, 1))
        object_stats = torch.zeros((self.num_entities, 1))
        for (s,p,o) in self.train:
            subject_stats[s] += 1
            relation_stats[p] += 1
            object_stats[o] += 1
        result = dict()
        for arg, stats, num in [
            ('subject', [i for i,j in list(sorted(enumerate(subject_stats.tolist()), key=lambda x:x[1]))], self.num_entities),
            ('relation', [i for i,j in list(sorted(enumerate(relation_stats.tolist()), key=lambda x:x[1]))], self.num_relations),
            ('object', [i for i,j in list(sorted(enumerate(object_stats.tolist()), key=lambda x:x[1]))], self.num_entities),
        ]:
            for percentile, (begin, end) in [('25%', (0., 0.25)), ('50%', (0.25, 0.5)), ('75%', (0.5, 0.75)), ('top', (0.75, 1.))]:
                if arg not in result:
                    result[arg] = dict()
                result[arg][percentile] = set(stats[int(begin*num):int(end*num)])
        self.indexes["frequency_percentiles"] = result
