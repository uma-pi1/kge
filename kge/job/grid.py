import copy
import os
from kge.job import Job
from kge import Config
import itertools


class GridJob(Job):
    """Job to perform grid search.

    This job creates a :class:`SearchJob` with one configuration for each point
    on the grid.

    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

    def run(self):
        # read grid search options range
        all_keys = []
        all_values = []
        all_indexes = []
        grid_configs = self.config.get("grid.configurations")
        for k, v in Config.flatten(grid_configs).items():
            all_keys.append(k)
            all_values.append(v)
            all_indexes.append(range(len(v)))

        # create search configs
        search_configs = []
        for indexes in itertools.product(*all_indexes):
            # obtain values for changed parameters
            values = list(
                map(lambda ik: all_values[ik[0]][ik[1]], enumerate(list(indexes)))
            )

            # create search configuration and check whether correct
            dummy_config = self.config.clone()
            search_config = Config(load_default=False)
            search_config.options["folder"] = "_".join(map(str, values))
            for i, key in enumerate(all_keys):
                dummy_config.set(key, values[i])  # to test whether correct k/v pair
                search_config.set(key, values[i], create=True)

            # and remember it
            search_configs.append(search_config.options)

        # create configuration file of search job
        self.config.set("job.type", "search")
        self.config.set("search.configurations", search_configs)
        self.config.save(os.path.join(self.config.folder, "config.yaml"))

        # and run it
        if self.config.get("grid.run"):
            job = Job.create(self.config, self.dataset, parent_job=self)
            job.resume()
            job.run()
        else:
            self.config.log("Skipping running of search job as requested by user...")
