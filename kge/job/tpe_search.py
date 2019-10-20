import concurrent.futures
import numpy as np
from kge.job import AutoSearchJob
from kge import Config
import kge.job.search
from hyperopt import fmin, Trials, tpe, hp, STATUS_OK


class TPESearchJob(AutoSearchJob):
    """
    Job for hyperparameter search using random search
    Source: http://hyperopt.github.io/hyperopt/
    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.trials = None
        self.space = None
        self.trial_no = None

    def init_search(self):
        self.trials = Trials()
        self.space = {}
        self.trial_no = 0

        # Read the parameters
        for i in self.config.get("tpe_search.parameters"):

            if i['type'] == "range":
                if "integer_space" in i.keys():
                    self.space.update({i['name']: hp.quniform(i['name'],
                                                              i['bounds'][0],
                                                              i['bounds'][1],
                                                              i['integer_space'])}
                                      )
                elif "log_scale" in i.keys() and i['log_scale']:
                    self.space.update({i['name']: hp.loguniform(i['name'],
                                                                np.log(i['bounds'][0]),
                                                                np.log(i['bounds'][1]))}
                                      )
                else:
                    self.space.update({i['name']: hp.uniform(i['name'],
                                                             i['bounds'][0],
                                                             i['bounds'][1])}
                                      )
            elif i['type'] == "choice":
                self.space.update({i['name']: hp.choice(i['name'],
                                                        i['values'])}
                                  )
            elif i['type'] == "fixed":
                self.space.update({i['name']: i['value']})
            else:
                raise ValueError("Unknown variable type")

    def register_trial(self, parameters=None):
        # Hyperopt does this itself
        return None, None

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # Hyperopt does this itself
        pass

    def get_best_parameters(self):
        # Hyperopt does this itself
        return None, None

    def objective(self, x):
        trial_no = self.trial_no
        for i in self.config.get("tpe_search.parameters"):
            if i['type'] == "range":
                if "integer_space" in i.keys():
                    x[i['name']] = int(x[i['name']])
                else:
                    x[i['name']] = float(x[i['name']])

        parameters = x

        # create job for trial
        folder = str("{:05d}".format(trial_no))
        config = self.config.clone(folder)
        config.set("job.type", "train")
        config.set_all(parameters)
        config.init_folder()

        # Run trial
        best = kge.job.search._run_train_job((
            self,
            trial_no,
            config,
            self.config.get("tpe_search.num_train_trials"),
            list(parameters.keys()),
        ))
        self.wait_task(concurrent.futures.ALL_COMPLETED)

        self.trial_no = trial_no + 1

        return {'loss': best[1]['metric_value'], 'status': STATUS_OK}

    def run(self):
        self.init_search()

        # Run TPE optimization program
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.config.get("tpe_search.max_evals"),
            trials=self.trials
        )

        self.config.log(
            "Best trial: metric_value: {}, parameters: {}".format(
                min(self.trials.losses()),
                best
            )
        )
