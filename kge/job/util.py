import torch
from torch import Tensor


def get_sp_po_coords_from_spo_batch(
    batch: Tensor, num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    num_ones = 0
    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        num_ones += len(sp_index.get((s, p), NOTHING))
        num_ones += len(po_index.get((p, o), NOTHING))

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()

        objects = sp_index.get((s, p), NOTHING)
        coords[current_index : (current_index + len(objects)), 0] = i
        coords[current_index : (current_index + len(objects)), 1] = objects
        current_index += len(objects)

        subjects = po_index.get((p, o), NOTHING) + num_entities
        coords[current_index : (current_index + len(subjects)), 0] = i
        coords[current_index : (current_index + len(subjects)), 1] = subjects
        current_index += len(subjects)

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0
):
    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )
    return labels
