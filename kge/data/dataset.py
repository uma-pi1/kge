import csv
import torch

class Dataset:
    def __init__(self,
                 config,
                 num_entities, entities):
        self.config = config
        self.num_entities = num_entities
        self.entities = entities

    def load(config):
        name = config.raw['dataset']['name']
        config.log('Loading ' + name)
        basedir = "data/" + name + "/"

        num_entities, entities = _load_map( basedir + config.raw['dataset']['entity_map'] )
        config.log(str(num_entities) + " entities")
        num_relations, relations = _load_map( basedir + config.raw['dataset']['relation_map'] )
        config.log(str(num_relations) + " relations")
