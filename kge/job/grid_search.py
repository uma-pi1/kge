import os
from kge.job import Job
from kge import Config
import itertools


class GridSearchJob(Job):
    """Job to perform grid search.

    This job creates a :class:`ManualSearchJob` with one configuration for each point on
    the grid.

    """

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        if self.__class__ == GridSearchJob:
            for f in Job.job_created_hooks:
                f(self)

    def _run(self):
        # read grid search options range
        all_keys = []
        all_keys_short = []
        all_values = []
        all_indexes = []
        grid_configs = self.config.get("grid_search.parameters")
        for k, v in sorted(Config.flatten(grid_configs).items()):
            all_keys.append(k)
            short_key = k[k.rfind(".") + 1 :]
            if "_" in short_key:
                # just keep first letter after each _
                all_keys_short.append(
                    "".join(map(lambda s: s[0], short_key.split("_")))
                )
            else:
                # keep up to three letters
                all_keys_short.append(short_key[:3])
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
            search_config.options["folder"] = "_".join(
                map(lambda i: all_keys_short[i] + str(values[i]), range(len(values)))
            )
            for i, key in enumerate(all_keys):
                dummy_config.set(key, values[i])  # to test whether correct k/v pair
                search_config.set(key, values[i], create=True)

            # and remember it
            search_configs.append(search_config.options)

        # create configuration file of search job
        self.config.set("search.type", "manual")
        self.config.set("manual_search.configurations", search_configs)
        self.config.save(os.path.join(self.config.folder, "config.yaml"))

        # and run it
        if self.config.get("grid_search.run"):
            job = Job.create(self.config, self.dataset, parent_job=self)
            job.run()
        else:
            self.config.log("Skipping running of search job as requested by user...")
