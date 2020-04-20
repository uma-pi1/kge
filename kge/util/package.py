import os
import torch
from kge import Config, Dataset


def add_package_parser(subparsers):
    """Creates the parser for the command package"""
    package_parser = subparsers.add_parser(
        "package",
        help="Create packaged model (checkpoint only containing model)",
    )
    package_parser.add_argument("checkpoint", type=str)


def package_model(checkpoint_path):
    """
    Converts a checkpoint to a packaged model.
    A packaged model only contains the model, entity/relation ids and the config.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    original_config = checkpoint["config"]
    config = Config()  # round trip to handle deprecated configs
    config.load_options(original_config.options)
    dataset = Dataset.load(config, preload_data=False)
    packaged_model = {
        "type": "package",
        "model": checkpoint["model"],
        "config": checkpoint["config"],
        "dataset_meta": dataset.save_meta(["entity_ids", "relation_ids", "entity_strings", "relation_strings"])
    }
    output_folder = os.path.dirname(checkpoint_path)
    filename = "model_{}.pt".format(checkpoint["epoch"])
    torch.save(packaged_model, os.path.join(output_folder, filename))
