import torch


def get_batch_sp_po_coords(
    batch, num_entities, sp_index: dict, po_index: dict
) -> torch.LongTensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entities columns.
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


def coord_to_sparse_tensor(nrows, ncols, coords, device, value=1.0):
    """Returns a sparse nrows x ncols tensor of labels from coordinates.

    Commonly, nrows denotes the batch size and ncols denotes the overall
    number of entities in a graph and coords holds batch indexes at the first
    column and entity-ids at the second column.

    """
    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )
    return labels
