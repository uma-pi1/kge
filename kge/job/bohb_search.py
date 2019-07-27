import concurrent.futures
from kge.job import AutoSearchJob
from kge import Config
import kge.job.search
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.core.worker import Worker


class BOHBSearchJob(AutoSearchJob):
    """
        Job for hyperparameter search using BOHB (Falkner et al. 2018)
        Source: https://github.com/automl/HpBandSter
    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.name_server = None
        self.worker = None

        # TODO: num_workers see example_2_local_parallel_threads

    def init_search(self):

        port = None \
            if self.config.get("bohb_search.port") == "None" \
            else self.config.get("bohb_search.port")

        self.name_server = hpns.NameServer(run_id=self.config.get("bohb_search.run_id"),
                                           host=self.config.get("bohb_search.host"),
                                           port=port)
        self.name_server.start()

        self.worker = BOHBWorker(
            sleep_interval=self.config.get("bohb_search.sleep_interval"),
            nameserver=self.config.get("bohb_search.host"),
            run_id=self.config.get("bohb_search.run_id"),
            job_config=self.config,
            parent_job=self
        )
        self.worker.run(background=True)

    def register_trial(self, parameters=None):
        # HyperBand does this itself
        pass

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # HyperBand does this itself
        pass

    def get_best_parameters(self):
        # HyperBand does this itself
        pass

    def run(self):
        self.init_search()

        bohb = BOHB(
            configspace=self.worker.get_configspace(self.config),
            run_id=self.config.get("bohb_search.run_id"),
            nameserver=self.config.get("bohb_search.host"),
            min_budget=self.config.get("bohb_search.min_budget"),
            max_budget=self.config.get("bohb_search.max_budget")
        )
        res = bohb.run(n_iterations=self.config.get("bohb_search.n_iter"))

        bohb.shutdown(shutdown_workers=True)
        self.name_server.shutdown()

        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()

        # TODO: Print best loss
        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                    sum([r.budget for r in res.get_all_runs()]) / self.config.get("bohb_search.max_budget")))


class BOHBWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        self.job_config = kwargs.pop('job_config')
        self.parent_job = kwargs.pop('parent_job')
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval
        self.next_trial_no = 0

    def compute(self, config, budget, **kwargs):
        """
        TODO
        config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        :param config:
        :param budget:
        :param kwargs:
        :return:
        """
        trial_no = self.next_trial_no

        parameters = config

        # create job for trial
        folder = str("{:05d}".format(trial_no))
        conf = self.job_config.clone(folder)
        conf.set("job.type", "train")
        conf.set_all(parameters)
        conf.init_folder()

        best = kge.job.search._run_train_job((
            self.parent_job,
            trial_no,
            conf,
            self.job_config.get("bohb_search.num_train_trials"),
            list(parameters.keys()),
        ))
        self.parent_job.wait_task(concurrent.futures.ALL_COMPLETED)

        self.next_trial_no = trial_no + 1

        return {'loss': best[1]['metric_value'], 'info': {}}

    @staticmethod
    def get_configspace(config):
        config_space = CS.ConfigurationSpace()

        parameters = config.get("bohb_search.parameters")
        for p in parameters:
            v_name = p['name']
            v_type = p['type']

            if v_type == 'categorical':
                config_space.add_hyperparameter(CSH.CategoricalHyperparameter(
                    v_name,
                    choices=p['choices']
                ))
            elif v_type == 'ordinal':
                config_space.add_hyperparameter(CSH.OrdinalHyperparameter(
                    v_name,
                    sequence=p['values']
                ))
            elif v_type == 'integer':
                if p['uniform']:
                    config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                        name=v_name,
                        lower=p['bounds'][0],
                        upper=p['bounds'][1],
                        log=p['log_scale']
                    ))
                else:
                    config_space.add_hyperparameter(CSH.NormalIntegerHyperparameter(
                        name=v_name,
                        mu=p['mu'],
                        sigma=p['sigma'],
                        log=p['log_scale']
                    ))
            elif v_type == 'float':
                if p['uniform']:
                    config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        v_name,
                        lower=p['bounds'][0],
                        upper=p['bounds'][1],
                        log=p['log_scale']
                    ))
                else:
                    config_space.add_hyperparameter(CSH.NormalFloatHyperparameter(
                        v_name,
                        mu=p['mu'],
                        sigma=p['sigma'],
                        log=p['log_scale']
                    ))
            else:
                # TODO: Raise error
                pass
        return config_space
