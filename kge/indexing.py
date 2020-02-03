from typing import Any, Dict, Tuple

import torch
from collections import defaultdict, OrderedDict


class Indexing:
    """Functions that compute and add indexes as needed; arguments is dataset.

    Index means any data structure that is derived from the dataset, including
    statistics and indexes.

    If the index has not yet been computed, computes it by calling the function
    specified in `self.index_functions`.

    """
    def __init__(self, dataset: "Dataset"):
        #: the parent dataset
        self.dataset : "Dataset" = dataset
        #: data derived automatically from the splits or meta data. Indexed by key.
        self._indexes: Dict[str, Any] = {}

    def __call__(self, index_name) -> Any:
        """Return the index stored under the specified name."""
        if index_name not in self._indexes:
            # either lookup class method with name 'index_name' or call
            # self.all_non_empty_fibers with argument index_name to compute
            # index.
            def all_non_empty_fibers():
                return self.all_non_empty_fibers(index_name)
            getattr(self, index_name, all_non_empty_fibers)()
        return self._indexes[index_name]

    @staticmethod
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

    def all_non_empty_fibers(self, split_coord_to_value: str) -> OrderedDict:
        """Return an index of all non empty fibers of a given mode of the
        ''train'', ''valid'' or ''test'' data tensor. A fiber contains all
        non-zero ''o'', ''s'' or ''p'' indexes for a mode. A mode is a fixed
        ''sp'', ''po'' or ''so'' coordinate, respectively.

        The index maps from `tuple' to `torch.LongTensor` and is cached under
        split_coord_to_value.

        The string split_coord_to_value is f.ex. ''train_sp_to_o''.

        """
        split, key, _, value = split_coord_to_value.split("_")
        triples = self.dataset.load_triples(split)
        if key == "sp":
            key_cols = [0, 1]
            value_column = 2
        elif key == "po":
            key_cols = [1, 2]
            value_column = 0
        elif key == "so":
            key_cols = [0, 2]
            value_column = 1
        else:
            raise ValueError()

        if not self._indexes.get(split_coord_to_value):
            self._indexes[split_coord_to_value] = self._group_by(
                triples[:, key_cols], triples[:, value_column]
            )
            self.dataset.config.log(
                "{} distinct {} pairs in {}".format(
                    len(self._indexes[split_coord_to_value]), key, split
                ),
                prefix="  ",
            )

        return self._indexes.get(split_coord_to_value)

    @staticmethod
    def prepare_index(index: OrderedDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert `all_non_empty_fibers` indexes to.

        Returns an nx2 keys tensor (rows = keys), an offset vector
        (row = starting offset in values for corresponding key),
        a values vector (entries correspond to values of original
        index)

        Afterwards, it holds:
            index[keys[i]] = values[offsets[i]:offsets[i+1]]
        """
        coord = torch.tensor(list(index.keys()), dtype=torch.int)
        values = torch.cat(list(index.values()))
        offsets = torch.cumsum(
            torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
        )
        return coord, values, offsets

    def _get_relation_types(self,) -> Dict:
        """
        Classify relations into 1-N, M-1, 1-1, M-N

        Bordes, Antoine, et al.
        "Translating embeddings for modeling multi-relational data."
        Advances in neural information processing systems. 2013.

        :return: dictionary mapping from int -> {1-N, M-1, 1-1, M-N}
        """
        relation_stats = torch.zeros((self.dataset.num_relations(), 6))
        for index, p in [
            (self.dataset.index("train_sp_to_o"), 1),
            (self.dataset.index("train_po_to_s"), 0),
        ]:
            for prefix, labels in index.items():
                relation_stats[prefix[p], 0 + p * 2] = labels.float().sum()
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5
        result = dict()
        for i, relation in enumerate(self.dataset.relation_ids()):
            result[i] = "{}-{}".format(
                "1" if relation_stats[i, 4].item() == 0 else "M",
                "1" if relation_stats[i, 5].item() == 0 else "N",
            )
        return result

    def relations_per_type(self):
        self.relation_types()

    def relation_types(self):
        """
        create dictionary mapping from {1-N, M-1, 1-1, M-N} -> set of relations
        """
        if "relation_types" in self._indexes:
            return
        relation_types = self._get_relation_types()
        relations_per_type = {}
        for k, v in relation_types.items():
            relations_per_type.setdefault(v, set()).add(k)
        for k, v in relations_per_type.items():
            self.dataset.config.log(
                "{} relations of type {}".format(len(v), k), prefix="  "
            )
        self._indexes["relation_types"] = relation_types
        self._indexes["relations_per_type"] = relations_per_type

    def frequency_percentiles(self, recompute=False):
        """
        Computes a dictionary mapping from
        {
            'subject':
            {25%, 50%, 75%, top} -> set of entities
            'relations':
            {25%, 50%, 75%, top} -> set of relations
            'object':
            {25%, 50%, 75%, top} -> set of entities
        }
        """
        if "frequency_percentiles" in self._indexes and not recompute:
            return
        subject_stats = torch.zeros((self.dataset.num_entities(), 1))
        relation_stats = torch.zeros((self.dataset.num_relations(), 1))
        object_stats = torch.zeros((self.dataset.num_entities(), 1))
        for (s, p, o) in self.dataset.train():
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
                self.dataset.num_entities(),
            ),
            (
                "relation",
                [
                    i
                    for i, j in list(
                        sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                    )
                ],
                self.dataset.num_relations(),
            ),
            (
                "object",
                [
                    i
                    for i, j in list(
                        sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                    )
                ],
                self.dataset.num_entities(),
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
        self._indexes["frequency_percentiles"] = result

    def _invert_ids(self, obj: str):
        if not f"{obj}_id_to_index" in self._indexes:
            ids = self.dataset.load_map(f"{obj}_ids")
            inv = {v: k for k, v in enumerate(ids)}
            self.dataset.config.log(f"Indexed {len(inv)} {obj} ids")
            self._indexes[f"{obj}_id_to_index"] = inv

    def entity_id_to_index(self):
        return self._invert_ids("entity")

    def relation_id_to_index(self):
        return self._invert_ids("relation")
