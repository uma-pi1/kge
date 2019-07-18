import csv
import os
import torch

from kge.util.misc import kge_base_dir


# TODO add support to pickle dataset (and indexes) and reload from there
class Dataset:
    """Class to load a specified dataset and process the data to create data objects
    """
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
        """
        Defines objects for entities and relations in test and train data from a dataset folder that contains data files
        Outputs the created objects (numbers) as Log, when running a Job.
        Returns the created data objects
        """
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
        """
        Takes a file with entity or relation maps as Input
        Returns the no. of entities/relations as an integer
        and the entities/relations as an array of strings
        """
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
        """
        Takes a file with train/test/val triples as input
        Returns a 2-way-tensor (nx3) with SPO triples
        and the triple row number as an array of strings
        """
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

    def index_1toN(self, what: str, key: str):
        """Return an index for the triples in what (''train'', ''valid'' or ''test'')
from the specified constituents (key: ''sp'' or ''po'') to the indexes of the
remaining constituent (''o'' or ''s'', respectively.)

        The index maps from `tuple' to `torch.LongTensor`.

        The index is cached in the provided dataset under name ''what_key''. If
        this index is already present, does not recompute it.

        """

        # Create dataset object
        if what == "train":
            triples = self.train
        elif what == "valid":
            triples = self.valid
        elif what == "test":
            triples = self.test
        else:
            raise ValueError()

        # Create key (SP or PO) object
        if key == "sp":
            key_columns = [0, 1]
            value_column = 2
        elif key == "po":
            key_columns = [1, 2]
            value_column = 0
        else:
            raise ValueError()

        # Create object for Dictionary SP: O or PO:S, save it as indexes in the Job, print no. of distinct tuples
        name = what + "_" + key
        if not self.indexes.get(name):
            index = Dataset._create_index_1toN(
                triples[:, key_columns], triples[:, value_column]
            )
            self.indexes[name] = index
            self.config.log(
                "{} distinct {} pairs in {}".format(len(index), key, what), prefix="  "
            )

        return self.indexes.get(name)

    @staticmethod
    def _create_index_1toN(key, value) -> dict:
        """Input: Keys:SP/PO tuples and corresponding values: P/O entities
        Output: Dictionary: SP: O or PO:S"""

        # Create dictionary
        result = {}
        for i in range(len(key)):
            k = (key[i, 0].item(), key[i, 1].item())
            values = result.get(k)
            if values is None:
                values = []
                result[k] = values
            values.append(value[i].item())
        # Make a (1-way) tensor out of the corresponding entity vector
        for key in result:
            result[key] = torch.LongTensor(sorted(result[key]))
        return result
