import os
import torch


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
    packaged_model = {
        "type": "package",
        "model": checkpoint["model"],
        "config": checkpoint["config"],
        "entity_ids": checkpoint["entity_ids"],
        "relation_ids": checkpoint["relation_ids"],
    }
    output_folder = os.path.dirname(checkpoint_path)
    filename = "model_{}.pt".format(checkpoint["epoch"])
    torch.save(packaged_model, os.path.join(output_folder, filename))
