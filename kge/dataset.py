import csv
import torch


# TODO add support to pickle dataset (and indexes) and reload from there
class Dataset:
    def __init__(self,
                 config,
                 num_entities, entities,
                 num_relations, relations,
                 train, train_meta,
                 valid, valid_meta,
                 test, test_meta):
        self.config = config
        self.num_entities = num_entities
        self.entities = entities  # array: entity index -> metadata array of strings
        self.num_relations = num_relations
        self.relations = relations  # array: relation index -> metadata array of strings
        self.train = train  # (n,3) int32 tensor
        self.train_meta = train_meta  # array: triple row number -> metadata array of strings
        self.valid = valid  # (n,3) int32 tensor
        self.valid_meta = valid_meta  # array: triple row number -> metadata array of strings
        self.test = test  # (n,3) int32 tensor
        self.test_meta = test_meta  # array: triple row number -> metadata array of strings
        self.indexes = {}  # map: name of index -> index (used mainly by training jobs)

    def load(config):
        name = config.get('dataset.name')
        config.log('Loading dataset ' + name + '...')
        basedir = "data/" + name + "/"

        num_entities, entities = Dataset._load_map(basedir + config.get('dataset.entity_map'))
        config.log(str(num_entities) + " entities", prefix='  ')
        num_relations, relations = Dataset._load_map(basedir + config.get('dataset.relation_map'))
        config.log(str(num_relations) + " relations", prefix='  ')

        train, train_meta = Dataset._load_triples(basedir + config.get('dataset.train'))
        config.log(str(len(train)) + " training triples", prefix='  ')

        valid, valid_meta = Dataset._load_triples(basedir + config.get('dataset.valid'))
        config.log(str(len(valid)) + " validation triples", prefix='  ')

        test, test_meta = Dataset._load_triples(basedir + config.get('dataset.test'))
        config.log(str(len(test)) + " test triples", prefix='  ')

        return Dataset(config, num_entities, entities, num_relations, relations,
                                     train, train_meta, valid, valid_meta, test, test_meta)

    def _load_map(filename):
        n = 0
        dictionary = {}
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                index = int(row[0])
                meta = row[1:]
                dictionary[index] = meta
                n = max(n, index + 1)
        array = [[]] * n
        for index, meta in dictionary.items():
            array[index] = meta
        return n, array

    def _load_triples(filename):
        n = 0
        dictionary = {}
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
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
        """Return an index for the triples in what (''train'', ''valid'', ''test'')
from the specified constituents (''sp'' or ''po'') to the indexes of the
remaining constituent (''o'' or ''s'', respectively.)

        The index maps from `tuple' to `torch.LongTensor`.

        The index is cached in the provided dataset under name ''what_key''. If
        this index is already present, does not recompute it.

        """
        if what == 'train':
            triples = self.train
        elif what == 'valid':
            triples = self.valid
        elif what == 'test':
            triples = self.test
        else:
            raise ValueError()

        if key == 'sp':
            key_columns = [0, 1]
            value_column = 2
        elif key == 'po':
            key_columns = [1, 2]
            value_column = 0
        else:
            raise ValueError()

        name = what + '_' + key
        if not self.indexes.get(name):
            index = Dataset._create_index_1toN(
                triples[:, key_columns], triples[:, value_column])
            self.indexes[name] = index
            self.config.log("{} distinct {} pairs in {}".format(
                len(index), key, what), prefix='  ')

        return self.indexes.get(name)

    def _create_index_1toN(key, value) -> dict:
        result = {}
        for i in range(len(key)):
            k = (key[i, 0].item(), key[i, 1].item())
            values = result.get(k)
            if values is None:
                values = []
                result[k] = values
            values.append(value[i].item())
        for key in result:
            result[key] = torch.LongTensor(sorted(result[key]))
        return result
