import os
import torch
from kge import Config
from kge.misc import is_number


def get_checkpoint_file(config: Config, checkpoint_argument: str = "default"):
    """
    Gets the path to a checkpoint file based on a config.

    Args:
        config: config specifying the folder
        checkpoint_argument: Which checkpoint to use: 'default', 'last', 'best',
                             a number or a file name

    Returns:
        path to a checkpoint file
    """
    if checkpoint_argument == "default":
        if config.get("job.type") in ["eval", "valid"]:
            checkpoint_file = config.checkpoint_file("best")
        else:
            last_epoch = config.last_checkpoint_number()
            if last_epoch is None:
                checkpoint_file = None
            else:
                checkpoint_file = config.checkpoint_file(last_epoch)
    elif is_number(checkpoint_argument, int) or checkpoint_argument == "best":
        checkpoint_file = config.checkpoint_file(checkpoint_argument)
    else:
        # otherwise, treat it as a filename
        checkpoint_file = checkpoint_argument
    return checkpoint_file


def load_checkpoint(checkpoint_file: str, device="cpu"):
    if not os.path.exists(checkpoint_file):
        raise IOError(
            "Specified checkpoint file {} does not exist.".format(checkpoint_file)
        )
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if device is not None and "config" in checkpoint:
        checkpoint["config"].set("job.device", device)
    checkpoint["file"] = checkpoint_file
    checkpoint["folder"] = os.path.dirname(checkpoint_file)
    return checkpoint
