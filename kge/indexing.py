import torch
from collections import defaultdict, OrderedDict


def _group_by(keys, values) -> dict:
    """ Groups values by keys.

    :param keys: list of keys
    :param values: list of values
    A key value pair i is defined by (key_list[i], value_list[i]).
    :return: OrderedDict where key value pairs have been grouped by key.

     """
    result = defaultdict(list)
    for key, value in zip(keys.tolist(), values.tolist()):
        result[tuple(key)].append(value)
    for key, value in result.items():
        result[key] = torch.IntTensor(sorted(value))
    return OrderedDict(result)


def index_KvsAll(dataset: "Dataset", split: str, key: str):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `split_sp` or `split_po` or
    `split_so`. If this index is already present, does not recompute it.

    """
    triples = dataset.split(split)
    value = None
    if key == "sp":
        key_cols = [0, 1]
        value_column = 2
        value = "o"
    elif key == "po":
        key_cols = [1, 2]
        value_column = 0
        value = "s"
    elif key == "so":
        key_cols = [0, 2]
        value_column = 1
        value = "p"
    else:
        raise ValueError()

    name = split + "_" + key + "_to_" + value
    if not dataset._indexes.get(name):
        dataset._indexes[name] = _group_by(
            triples[:, key_cols], triples[:, value_column]
        )
        dataset.config.log(
            "{} distinct {} pairs in {}".format(
                len(dataset._indexes[name]), key, split
            ),
            prefix="  ",
        )

    return dataset._indexes.get(name)


def prepare_index(index):
    """Convert `index_KvsAll` indexes to pytorch tensors.

    Returns an nx2 keys tensor (rows = keys), an offset vector
    (row = starting offset in values for corresponding key),
    a values vector (entries correspond to values of original
    index)

    Afterwards, it holds:
        index[keys[i]] = values[offsets[i]:offsets[i+1]]
    """
    sp_po = torch.tensor(list(index.keys()), dtype=torch.int)
    o_s = torch.cat(list(index.values()))
    offsets = torch.cumsum(
        torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
    )
    return sp_po, o_s, offsets


def _get_relation_types(dataset,):
    """
    Classify relations into 1-N, M-1, 1-1, M-N

    Bordes, Antoine, et al.
    "Translating embeddings for modeling multi-relational data."
    Advances in neural information processing systems. 2013.

    :return: dictionary mapping from int -> {1-N, M-1, 1-1, M-N}
    """
    relation_stats = torch.zeros((dataset.num_relations(), 6))
    for index, p in [
        (dataset.index("train_sp_to_o"), 1),
        (dataset.index("train_po_to_s"), 0),
    ]:
        for prefix, labels in index.items():
            relation_stats[prefix[p], 0 + p * 2] = labels.float().sum()
            relation_stats[prefix[p], 1 + p * 2] = (
                relation_stats[prefix[p], 1 + p * 2] + 1.0
            )
    relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
    relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
    result = dict()
    for i, relation in enumerate(dataset.relation_ids()):
        result[i] = "{}-{}".format(
            "1" if relation_stats[i, 4].item() == 0 else "M",
            "1" if relation_stats[i, 5].item() == 0 else "N",
        )
    return result


def index_relation_types(dataset):
    """
    create dictionary mapping from {1-N, M-1, 1-1, M-N} -> set of relations
    """
    if "relation_types" in dataset._indexes:
        return
    relation_types = _get_relation_types(dataset)
    relations_per_type = {}
    for k, v in relation_types.items():
        relations_per_type.setdefault(v, set()).add(k)
    for k, v in relations_per_type.items():
        dataset.config.log("{} relations of type {}".format(len(v), k), prefix="  ")
    dataset._indexes["relation_types"] = relation_types
    dataset._indexes["relations_per_type"] = relations_per_type


def index_frequency_percentiles(dataset, recompute=False):
    """
    :return: dictionary mapping from
    {
        'subject':
        {25%, 50%, 75%, top} -> set of entities
        'relations':
        {25%, 50%, 75%, top} -> set of relations
        'object':
        {25%, 50%, 75%, top} -> set of entities
    }
    """
    if "frequency_percentiles" in dataset._indexes and not recompute:
        return
    subject_stats = torch.zeros((dataset.num_entities(), 1))
    relation_stats = torch.zeros((dataset.num_relations(), 1))
    object_stats = torch.zeros((dataset.num_entities(), 1))
    for (s, p, o) in dataset.train():
        subject_stats[s] += 1
        relation_stats[p] += 1
        object_stats[o] += 1
    result = dict()
    for arg, stats, num in [
        (
            "subject",
            [
                i
                for i, j in list(
                    sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
        (
            "relation",
            [
                i
                for i, j in list(
                    sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_relations(),
        ),
        (
            "object",
            [
                i
                for i, j in list(
                    sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
    ]:
        for percentile, (begin, end) in [
            ("25%", (0.0, 0.25)),
            ("50%", (0.25, 0.5)),
            ("75%", (0.5, 0.75)),
            ("top", (0.75, 1.0)),
        ]:
            if arg not in result:
                result[arg] = dict()
            result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
    dataset._indexes["frequency_percentiles"] = result


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset.config.log(f"Indexed {len(inv)} {obj} ids")
        dataset._indexes[f"{obj}_id_to_index"] = inv


def create_default_index_functions(dataset: "Dataset"):
    for split in ["train", "valid", "test"]:
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p")]:
            # self assignment needed to capture the loop var
            dataset.index_functions[
                f"{split}_{key}_to_{value}"
            ] = lambda dataset, split=split, key=key: index_KvsAll(
                dataset, split, key
            )
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relation_types
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles

    dataset.index_functions["entity_id_to_index"] = lambda dataset: _invert_ids(
        dataset, "entity"
    )
    dataset.index_functions["relation_id_to_index"] = lambda dataset: _invert_ids(
        dataset, "relation"
    )
