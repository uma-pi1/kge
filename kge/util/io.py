import os
import torch


def load_checkpoint(checkpoint_file: str, device="cpu"):
    if not os.path.exists(checkpoint_file):
        raise IOError(
            "Specified checkpoint file {} does not exist.".format(checkpoint_file)
        )
    checkpoint = torch.load(checkpoint_file, map_location=device)
    checkpoint["file"] = checkpoint_file
    checkpoint["folder"] = os.path.dirname(checkpoint_file)
    return checkpoint
