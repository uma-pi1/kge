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
        self.entities = entities              # array: entity index -> metadata array of strings
        self.num_relations = num_relations
        self.relations = relations            # array: relation index -> metadata array of strings
        self.train = train                    # (n,3) int32 tensor
        self.train_meta = train_meta          # array: triple row number -> metadata array of strings
        self.valid = valid                    # (n,3) int32 tensor
        self.valid_meta = valid_meta          # array: triple row number -> metadata array of strings
        self.test = test                      # (n,3) int32 tensor
        self.test_meta = test_meta            # array: triple row number -> metadata array of strings
        self.indexes = { }                    # map: name of index -> index (used mainly by training jobs)

    def load(config):
        name = config.get('dataset.name')
        config.log('Loading dataset ' + name)
        basedir = "data/" + name + "/"

        num_entities, entities = Dataset._load_map( basedir + config.get('dataset.entity_map') )
        config.log(str(num_entities) + " entities")
        num_relations, relations = Dataset._load_map( basedir + config.get('dataset.relation_map') )
        config.log(str(num_relations) + " relations")

        train, train_meta = Dataset._load_triples( basedir + config.get('dataset.train') )
        config.log(str(len(train)) + " training triples")

        valid, valid_meta = Dataset._load_triples( basedir + config.get('dataset.valid') )
        config.log(str(len(valid)) + " validation triples")

        test, test_meta = Dataset._load_triples( basedir + config.get('dataset.test') )
        config.log(str(len(test)) + " test triples")

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
                n = max(n, index+1)
        array = [ [] ] * n
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
                dictionary[n] = (torch.IntTensor([s,p,o]),meta)
                n += 1
        triples = torch.empty(n, 3, dtype=torch.int32)
        meta = [ [] ] * n
        for index, value in dictionary.items():
            triples[index,:] = value[0]
            meta[index] = value[1]
        return triples, meta
