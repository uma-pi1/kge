import os
from kge import Config
from kge.misc import kge_base_dir
from os import path
import shutil

def create_config(test_dataset_name: str, model: str = "complex") -> Config:
    config = Config()
    config.folder = None
    config.set("verbose", False)
    config.set("model", model)
    config._import(model)
    config.set("dataset.name", test_dataset_name)
    config.set("job.device", "cpu")
    return config


def get_dataset_folder(dataset_name):
    return os.path.join(kge_base_dir(), "tests", "data", dataset_name)


def get_cache_dir():
    return os.path.join(kge_base_dir(), "tests", "data", "cache")


def empty_cache():
    for file in os.listdir(get_cache_dir()):
        obj = path.join(get_cache_dir(), file)
        if os.path.isfile(obj):
            os.remove(obj)
        else:
            shutil.rmtree(obj)
