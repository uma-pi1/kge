import torch
from collections import defaultdict, OrderedDict
import numba
import numpy as np
from typing import Dict, List, Iterator, Tuple


class KvsAllIndex:
    """Construct an index from keys (e.g., sp) to all its values (o).

    Keys are tuples, values are PyTorch tensors.

    Internally stores list of unique keys, list of values, and starting offset of each
    key in values in PyTorch tensors. Access by key is enabled using an index on top of
    these tensors. The tensors can also be used directly (e.g., in KvsAll training)

    """

    def __init__(
        self,
        triples: torch.Tensor,
        key_cols: List,
        value_col: int,
        default_factory: type,
    ):
        """
        Args:
            triples: data
            key_cols: the two columns used as keys
            value_col: column used as value
            default_factory: default return type
        """
        self.key_cols = key_cols
        self.value_col = value_col

        # sort triples, extract unique keys, all values, and starting offset of each key
        triples_sorted = KvsAllIndex.sort_triples_by_keys(triples, key_cols, value_col)
        keys, values_offset = np.unique(
            triples_sorted[:, key_cols], axis=0, return_index=True
        )
        values_offset = np.append(values_offset, len(triples_sorted))

        # create disctionary from key (as a tuple) to key index
        self._index_of_key = dict()
        for key_index, key in enumerate(keys):
            self._index_of_key[tuple(key)] = key_index

        # convert data structures to pytorch and keep them
        self._keys = torch.from_numpy(keys)
        self._values_offset = torch.from_numpy(values_offset)
        self._values = triples_sorted[
            :, self.value_col
        ].clone()  # to drop reference to triples_sorted

        self.default_factory = default_factory

    def __getitem__(self, key, default_return_value=None) -> torch.Tensor:
        try:
            key_index = self._index_of_key[key]
            return self._values_of(key_index)
        except KeyError:
            if default_return_value is None:
                return self.default_factory()
            return default_return_value

    def _values_of(self, key_index) -> torch.Tensor:
        start = self._values_offset[key_index]
        end = self._values_offset[key_index + 1]
        return self._values[start:end]

    def __len__(self):
        return len(self._keys)

    def get(self, key, default_return_value=None) -> torch.Tensor:
        return self.__getitem__(key, default_return_value)

    def keys(self) -> Iterator[Tuple[int, int]]:
        return self._index_of_key.keys()

    def values(self) -> Iterator[torch.Tensor]:
        return [self._values_of(key_index) for key_index in self._index_of_key.values()]

    def items(self) -> Iterator[Tuple[Tuple[int, int], torch.Tensor]]:
        return zip(self.keys(), self.values())

    @staticmethod
    def sort_triples_by_keys(
        triples: torch.Tensor, key_cols: List, value_col: int
    ) -> torch.Tensor:
        """Sorts triples by key_cols, then value_col."""
        # using numpy, since torch has no stable sort
        triples = triples.numpy()
        triples_sorted = triples[np.argsort(triples[:, value_col])]
        for key in key_cols[::-1]:
            triples_sorted = triples_sorted[
                np.argsort(triples_sorted[:, key], kind="stable")
            ]
        return torch.from_numpy(triples_sorted)


def index_KvsAll(dataset: "Dataset", split: str, key: str):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `{split}_sp_to_o` or
    `{split}_po_to_s`, or `{split}_so_to_p`. If this index is already present, does not
    recompute it.

    """
    value = None
    if key == "sp":
        key_cols = [0, 1]
        value_col = 2
        value = "o"
    elif key == "po":
        key_cols = [1, 2]
        value_col = 0
        value = "s"
    elif key == "so":
        key_cols = [0, 2]
        value_col = 1
        value = "p"
    else:
        raise ValueError()

    name = split + "_" + key + "_to_" + value
    if not dataset._indexes.get(name):
        triples = dataset.split(split)
        dataset._indexes[name] = KvsAllIndex(triples, key_cols, value_col, list)

    dataset.config.log(
        "{} distinct {} pairs in {}".format(len(dataset._indexes[name]), key, split),
        prefix="  ",
    )

    return dataset._indexes.get(name)


def index_relation_types(dataset):
    """Classify relations into 1-N, M-1, 1-1, M-N.

    According to Bordes et al. "Translating embeddings for modeling multi-relational
    data.", NIPS13.

    Adds index `relation_types` with list that maps relation index to ("1-N", "M-1",
    "1-1", "M-N").

    """
    if "relation_types" not in dataset._indexes:
        # 2nd dim: num_s, num_distinct_po, num_o, num_distinct_so, is_M, is_N
        relation_stats = torch.zeros((dataset.num_relations(), 6))
        for index, p in [
            (dataset.index("train_sp_to_o"), 1),
            (dataset.index("train_po_to_s"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = relation_stats[
                    prefix[p], 0 + p * 2
                ] + len(labels)
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        relation_types = []
        for i in range(dataset.num_relations()):
            relation_types.append(
                "{}-{}".format(
                    "1" if relation_stats[i, 4].item() == 0 else "M",
                    "1" if relation_stats[i, 5].item() == 0 else "N",
                )
            )

        dataset._indexes["relation_types"] = relation_types

    return dataset._indexes["relation_types"]


def index_relations_per_type(dataset):
    if "relations_per_type" not in dataset._indexes:
        relations_per_type = {}
        for i, k in enumerate(dataset.index("relation_types")):
            relations_per_type.setdefault(k, set()).add(i)
        dataset._indexes["relations_per_type"] = relations_per_type
    else:
        relations_per_type = dataset._indexes["relations_per_type"]

    dataset.config.log("Loaded relation index")
    for k, relations in relations_per_type.items():
        dataset.config.log(
            "{} relations of type {}".format(len(relations), k), prefix="  "
        )

    return relations_per_type


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
    for (s, p, o) in dataset.split("train"):
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


class IndexWrapper:
    """Wraps a call to an index function so that it can be pickled"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def __call__(self, dataset: "Dataset", **kwargs):
        self.fun(dataset, **self.kwargs)


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset._indexes[f"{obj}_id_to_index"] = inv
    else:
        inv = dataset._indexes[f"{obj}_id_to_index"]
    dataset.config.log(f"Indexed {len(inv)} {obj} ids", prefix="  ")


def create_default_index_functions(dataset: "Dataset"):
    for split in dataset.files_of_type("triples"):
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p")]:
            # self assignment needed to capture the loop var
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key
            )
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relations_per_type
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles

    for obj in ["entity", "relation"]:
        dataset.index_functions[f"{obj}_id_to_index"] = IndexWrapper(
            _invert_ids, obj=obj
        )


@numba.njit
def where_in(x, y, not_in=False):
    """Retrieve the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: not_in: if True, returns the indices of the of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. Also: "i in y" raises an error in numba
    # setting njit(parallel=True) slows down the function
    list_y = set(y)
    return np.where(np.array([i in list_y for i in x]) != not_in)[0]
