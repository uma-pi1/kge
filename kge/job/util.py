import torch


def get_batch_sp_po_coords(batch, num_entities, sp_index, po_index) -> torch.LongTensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    num_ones = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        num_ones += len(sp_index[(s, p)])
        num_ones += len(po_index[(p, o)])

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()

        objects = sp_index[(s, p)]
        coords[current_index:(current_index+len(objects)), 0] = i
        coords[current_index:(current_index+len(objects)), 1] = objects
        current_index += len(objects)

        subjects = po_index[(p, o)] + num_entities
        coords[current_index:(current_index+len(subjects)), 0] = i
        coords[current_index:(current_index+len(subjects)), 1] = subjects
        current_index += len(subjects)

    return coords
