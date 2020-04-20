import os
import torch


def load_checkpoint(checkpoint_file: str, device="cpu"):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    checkpoint["file"] = checkpoint_file
    checkpoint["folder"] = os.path.dirname(checkpoint_file)
    return checkpoint
