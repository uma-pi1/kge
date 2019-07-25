import numpy as np
from kge.job import AutoSearchJob
from kge import Config
import kge.job.search
from hyperopt import fmin, Trials, rand, hp
import sys


class RandomSearchJob(AutoSearchJob):
    """Job for hyperparameter search using random search TODO Website"""

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.parameters = None
        self.trials = None
        self.space = None

    # Overridden such that instances of search job can be pickled to workers
    def __getstate__(self):
        state = super(RandomSearchJob, self).__getstate__()
        del state["random_client"]
        return state

    def init_search(self):
        self.parameters = {}
        self.trials = Trials()
        self.space = {}

        for i in self.config.get("random_search.parameters"):
            self.parameters.update({i['name']: i['bounds'][0]})

            if i['type'] == "range":
                if "integer_space" in i.keys():
                    self.space.update({i['name']: hp.quniform(i['name'],
                                                             i['bounds'][0],
                                                             i['bounds'][1],
                                                             i['integer_space'])})
                elif "log_scale" in i.keys() and i['log_scale']:
                    self.space.update({i['name']: hp.loguniform(i['name'],
                                                                np.log(i['bounds'][0]),
                                                                np.log(i['bounds'][1]))})
                else:
                    self.space.update({i['name']: hp.uniform(i['name'],
                                                             i['bounds'][0],
                                                             i['bounds'][1])})
            elif i['type'] == "choice":
                self.space.update({i['name']: i['values']})
            elif i['type'] == "fixed":
                self.space.update({i['name']: i['value']})
            else:
                # Raise error
                pass

    def register_trial(self, parameters=None):
        # Hyperopt does this itself
        return None, None

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # Hyperopt does this itself
        pass

    def get_best_parameters(self):
        best_parameters = sorted(self.trials.results, key=lambda x: x['loss'])
        return best_parameters[0], best_parameters[0]['loss']

    def run(self):
        self.init_search()
        trial_no = 0

        # create job for trial
        folder = str("{:05d}".format(trial_no))
        config = self.config.clone(folder)
        config.set("job.type", "train")
        config.set_all(self.parameters)
        config.init_folder()

        def objective(x):
            best = kge.job.search._run_train_job((
                self,
                trial_no,
                config,
                self.config.get("random_search.num_trials"),
                list(self.parameters.keys()),
            ))
            return best[2]

        best = fmin(
            fn=objective,
            space=self.space,
            algo=rand.suggest,
            max_evals=self.config.get("random_search.max_evals"),
            trials=self.trials
        )

        print(best)
        print("xxx")
        print(self.trials.trials)

        # # TODO develop own code to output results
        # # all done, output best trial result
        # trial_metric_values = list(
        #     map(lambda trial_best: trial_best["metric_value"], self.results)
        # )
        # best_trial_index = trial_metric_values.index(max(trial_metric_values))
        # metric_name = self.results[best_trial_index]["metric_name"]
        # self.config.log(
        #     "Best trial: {}={}".format(
        #         metric_name, trial_metric_values[best_trial_index]
        #     )
        # )
        # self.trace(echo=True,
        #            echo_prefix="  ",
        #            log=True,
        #            scope="search",
        #            **self.results[best_trial_index])
