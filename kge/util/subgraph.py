import os
import copy
import numba
import numpy as np
import igraph as ig
from kge import Config, Dataset
import yaml
import pandas as pd
from typing import Dict


class KCoreManager:
    """Describe and manage k-core subsets."""

    def __init__(self, dataset: Dataset):
        """
        Initialize the KCoreManager.
        :param dataset: (Dataset) original dataset for which to manage the k-core
        subsets
        """
        # store the original dataset
        self._dataset = dataset
        # dictionary for k-core dataset objects (using int value k as key)
        self._subsets: Dict[int, Dataset] = {}

    def get_k_core_dataset(self, k: int):
        """
        Returns a k-core dataset object that is either taken from self._subsets or will
        be loaded if not done yet. If the request k-core does not yet exist, the
        decomposition will automatically be performed.
        :param k: (int) k value of requested k-core subset
        :return: k-core Dataset object
        """
        if k == 0:
            # use the original dataset if k == 0
            return self._dataset
        elif k not in self._subsets:
            # create new dataset using the path to k-core
            dataset_name = os.path.basename(os.path.normpath(self._dataset.folder))
            k_core_name = os.path.join(dataset_name, "subsets", f"{k}-core")
            path_to_k_core = os.path.join(self._dataset.folder, "subsets", f"{k}-core")
            k_core_config = copy.deepcopy(self._dataset.config)
            # if the subset folder does not exist, perform the k-core decomposition and
            # try again
            if not os.path.exists(path_to_k_core):
                self._dataset.config.log(
                    f"The {k}-core of dataset {dataset_name} was not found. Will"
                    f"perform a k-core decomposition now"
                )
                self._perform_k_core_decomposition()
                if not os.path.exists(path_to_k_core):
                    raise ValueError(
                        f"The dataset {dataset_name} has no {k}-core"
                    )
            k_core_config.set("dataset.name", k_core_name)
            subset = self._dataset.create(k_core_config, folder=path_to_k_core,
                                          overwrite=Config.Overwrite.Yes)
            self._subsets[k] = subset
        return self._subsets[k]

    def get_k_core_path(self, k: int):
        """
        Returns the path to a k-core folder. Will perform a k-core decomposition if the
        requested k-core was not found. If the k-core is still not found afterwards, a
        ValueError will be raised.
        :param k: (int) k value of requested k-core subset
        :return: path to k-core subset
        """
        if k == 0:
            # return the original dataset path if k == 0
            return self._dataset.folder
        # construct the path
        path_to_k_core = os.path.join(self._dataset.folder, "subsets", f"{k}-core")
        if not os.path.exists(path_to_k_core):
            dataset_name = os.path.basename(os.path.normpath(self._dataset.folder))
            self._dataset.config.log(
                f"The {k}-core of dataset {dataset_name} was not found. Will"
                f"perform a k-core decomposition now"
            )
            self._perform_k_core_decomposition()
            if not os.path.exists(path_to_k_core):
                raise ValueError(
                    f"The dataset {dataset_name} has no {k}-core"
                )
        return path_to_k_core

    def get_k_core_stats(self):
        """
        Returns statistics about k-core subsets for a given dataset. If the statistics
        file is not yet available, the decomposition will automatically be performed.
        """
        try:
            path_to_stats = os.path.join(
                self._dataset.folder, "subsets", "subset_stats.yaml"
            )
            with open(
                path_to_stats, "r"
            ) as stream:
                subset_stats = yaml.safe_load(stream)
            self._dataset.config.log(
                f"Loaded subset statistics from {path_to_stats}"
            )
        except IOError:
            subset_stats = self._perform_k_core_decomposition()

        return subset_stats

    def _perform_k_core_decomposition(self):
        """
        Performs the k-core decomposition with the help of igraph (https://igraph.org/).
        Therefore, all entities and interrelations that are contained in the k-cores
        (for all k) of a given graph dataset are kept. The subgraphs get saved under
        <dataset_name>/subsets/<k>-core. Note that multiple parallel edges only count as
        1 in our current k-core decomposition implementation.
        """
        dataset_name = os.path.basename(os.path.normpath(self._dataset.folder))

        self._dataset.config.log(
            f"Started the k-core decomposition for {dataset_name}"
        )

        # initialize subset_stats dict
        subset_stats = dict()

        # take required parts from the dataset object
        train = self._dataset.split("train")
        num_entities = self._dataset.num_entities()

        # convert tensor to numpy array
        train_np = train.cpu().detach().numpy()

        # perform k-core decomposition
        vertices = np.unique(train_np[:, (0, 2)])
        edges = train_np[:, (0, 2)]

        # create igraph
        graph = ig.Graph()
        graph.add_vertices(vertices)
        graph.add_edges(edges)
        graph.simplify(multiple=True, loops=True)

        # compute core values
        core_numbers = graph.coreness()

        # add whole graph stats
        subset_stats[0] = {
            "train": len(train),
            "entities": num_entities,
            "rel_entities": 1.0,
            "triples": 1.0,
            "triples_and_entities": 1.0,
            "path_to_dataset": self._dataset.config.get("dataset.name"),
        }

        # compute k-cores
        k = 1
        previous_subset = train_np
        while True:
            # obtain entities of current k-core subgraph
            core_indices = [
                v_idx for v_idx in range(len(vertices)) if core_numbers[v_idx] >= k
            ]
            k_core_graph = graph.subgraph(core_indices)
            if k_core_graph.vcount() == 0:
                # exit loop if max k was reached
                break
            else:
                # select all triples that are contained in k-core
                v_selected = k_core_graph.get_vertex_dataframe().name.values

                # filter the previous subgraph triples with the list of entities
                subset_core_indices = self._numba_is_in_2d(previous_subset, v_selected)
                subset_core = previous_subset[subset_core_indices]
                previous_subset = subset_core.copy()

                # finalize all files and compute stats
                (subset_files, subset_stats) = self._finalize_and_compute_stats(
                    subset_core, subset_stats, k
                )

                self._dataset.config.log(
                    f"Computed the {k}-core of {dataset_name} " 
                    f"({subset_stats[k]['entities']} entities, " 
                    f"{subset_stats[k]['train']} triples)"
                )

                # save subset files
                self._save_subset(subset_files, k)

                k += 1

        # write subset stats file
        with open(
            os.path.join(self._dataset.folder, "subsets", "subset_stats.yaml"), "w+"
        ) as filename:
            filename.write(yaml.dump(subset_stats))

        self._dataset.config.log("Finished the k-core decomposition")

        return subset_stats

    def _finalize_and_compute_stats(self, subset, subset_stats, k):
        """
        Finalize the subset creation and compute subset statistics.
        """

        # filter and reindex files
        entities, relations, subset = self._filter_entities_relations(subset)

        # perform train-valid-split
        train, valid = self._train_valid_split(subset)

        # compute relative triples and entities compared to original graph
        # these values correspond to the available cost metrics in the GraSH search config
        rel_triples = len(train) / subset_stats[0]["train"]
        rel_entities = len(entities) / subset_stats[0]["entities"]
        rel_triples_and_entities = rel_triples * rel_entities

        # add subset statistics to dict
        subset_stats[k] = {
            "train": len(train),
            "entities": len(entities),
            "rel_entities": rel_entities,
            "triples": rel_triples,
            "triples_and_entities": rel_triples_and_entities,
            "path_to_dataset": os.path.join(
                self._dataset.config.get("dataset.name"), "subsets", f"{k}-core"
            ),
        }

        subset_files = [entities, relations, train, valid]

        # add subset and filtered files to dict
        return subset_files, subset_stats

    def _train_valid_split(self, subset):
        """
        Randomly split the subset into train and valid sets w.r.t. max amount specified
        in the config.
        """

        # determine size of valid split - use defined fraction or max number if
        # use of fraction would exceed it.
        number_valid = min(
            [
                round(len(subset) * self._dataset.config.get(
                    "grash_search.valid_frac")),
                self._dataset.config.get("grash_search.valid_max"),
            ]
        )

        np.random.shuffle(subset)
        valid, train = subset[:number_valid, :], subset[number_valid:, :]

        return train, valid

    def _filter_entities_relations(self, subset):
        """
        Filter entities and relations and only keep those that appear in the subset.
        Also reindex entities and
        relations for the required density.
        """

        # take required parts from the dataset object
        all_entity_ids = np.arange(self._dataset.num_entities())
        all_entities = np.array(self._dataset.entity_strings())
        all_relation_ids = np.arange(self._dataset.num_relations())
        all_relations = np.array(self._dataset.relation_strings())

        # collect entities and relations from triple subset
        selected_entity_ids = np.unique(subset[:, (0, 2)])
        selected_relation_ids = np.unique(subset[:, 1])

        # only select entities and relations that appear in subset
        entities_indices = self._numba_is_in_1d(all_entity_ids, selected_entity_ids)
        entities = all_entities[entities_indices]
        relation_indices = self._numba_is_in_1d(all_relation_ids, selected_relation_ids)
        relations = all_relations[relation_indices]

        # reindex the entity and relation ids
        new_entity_ids = np.arange(len(entities))
        new_relation_ids = np.arange(len(relations))

        entity_mapper = np.empty(len(all_entity_ids), dtype=np.long)
        relation_mapper = np.empty(len(all_relation_ids), dtype=np.long)

        entity_mapper[selected_entity_ids] = new_entity_ids
        relation_mapper[selected_relation_ids] = new_relation_ids

        for (i, mapper) in [(0, entity_mapper), (1, relation_mapper),
                            (2, entity_mapper)]:
            subset[:, i] = mapper[subset[:, i]]

        entities_new = np.vstack((new_entity_ids, entities)).transpose()
        relations_new = np.vstack((new_relation_ids, relations)).transpose()

        return entities_new, relations_new, subset

    def _save_subset(self, subset_files, k):
        """
        Save subset files in del format into new subfolder with syntax
        <dataset_name>_<k>-core.
        """

        # check if subsets folder was already created
        if not os.path.exists(os.path.join(self._dataset.folder, "subsets")):
            os.mkdir(os.path.join(self._dataset.folder, "subsets"))

        # check if k_core folder was already created
        subset_folder = f"{k}-core"
        path_to_subset_folder = os.path.join(
            self._dataset.folder, "subsets", subset_folder
        )
        if not os.path.exists(path_to_subset_folder):
            os.mkdir(path_to_subset_folder)

        # save subset files
        pd.DataFrame(subset_files[0]).to_csv(
            os.path.join(path_to_subset_folder, "entity_ids.del"),
            sep="\t",
            header=False,
            index=False,
        )
        pd.DataFrame(subset_files[1]).to_csv(
            os.path.join(path_to_subset_folder, "relation_ids.del"),
            sep="\t",
            header=False,
            index=False,
        )
        pd.DataFrame(subset_files[2]).to_csv(
            os.path.join(path_to_subset_folder, "train.del"),
            sep="\t",
            header=False,
            index=False,
        )
        pd.DataFrame(subset_files[3]).to_csv(
            os.path.join(path_to_subset_folder, "valid.del"),
            sep="\t",
            header=False,
            index=False,
        )

        # create dataset.yaml - use valid file also as test file because no test set is
        # available nor required
        dataset_yaml = {
            "files.entity_ids.filename": "entity_ids.del",
            "files.entity_ids.type": "map",
            "files.relation_ids.filename": "relation_ids.del",
            "files.relation_ids.type": "map",
            "files.train.filename": "train.del",
            "files.train.size": len(subset_files[2]),
            "files.train.split_type": "train",
            "files.train.type": "triples",
            "files.valid.filename": "valid.del",
            "files.valid.size": len(subset_files[3]),
            "files.valid.split_type": "valid",
            "files.valid.type": "triples",
            "files.test.filename": "valid.del",
            "files.test.size": len(subset_files[3]),
            "files.test.split_type": "valid",
            "files.test.type": "triples",
            "name": os.path.join(
                self._dataset.config.get("dataset.name"), "subsets", subset_folder
            ),
            "num_entities": len(subset_files[0]),
            "num_relations": len(subset_files[1]),
        }

        # write dataset.yaml file
        with open(os.path.join(path_to_subset_folder, "dataset.yaml"), "w+") as filename:
            filename.write(yaml.dump(dict(dataset=dataset_yaml)))

    @staticmethod
    @numba.njit(parallel=True)
    def _numba_is_in_2d(arr, vec2):
        """
        Return filtering mask for values that occur in vec2.
        """

        out = np.empty(arr.shape[0], dtype=numba.boolean)
        vec2_set = set(vec2)

        for i in numba.prange(arr.shape[0]):
            if arr[i][0] in vec2_set and arr[i][2] in vec2_set:
                out[i] = True
            else:
                out[i] = False

        return out

    @staticmethod
    @numba.njit(parallel=True)
    def _numba_is_in_1d(arr, vec2):
        """
        Return filtering mask for values that occur in vec2.
        """

        out = np.empty(arr.shape[0], dtype=numba.boolean)
        vec2_set = set(vec2)

        for i in numba.prange(arr.shape[0]):
            if arr[i] in vec2_set:
                out[i] = True
            else:
                out[i] = False

        return out
