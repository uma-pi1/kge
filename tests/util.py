import os
from kge import Config
from kge.misc import kge_base_dir


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
