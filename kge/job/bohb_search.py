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
        self.name_server = None  # Server address to run the job on
        self.workers = []       # Workers that will run in parallel

    def init_search(self):
        # Assigning the port
        port = None \
            if self.config.get("bohb_search.port") == "None" \
            else self.config.get("bohb_search.port")

        # Assigning the address
        self.name_server = hpns.NameServer(run_id=self.config.get("bohb_search.run_id"),
                                           host=self.config.get("bohb_search.host"),
                                           port=port)
        # Start the server
        self.name_server.start()

        # Create workers
        for i in range(self.config.get("bohb_search.n_workers")):
            w = BOHBWorker(
                sleep_interval=self.config.get("bohb_search.sleep_interval"),
                nameserver=self.config.get("bohb_search.host"),
                run_id=self.config.get("bohb_search.run_id"),
                job_config=self.config,
                parent_job=self,
                id=i
            )
            w.run(background=True)
            self.workers.append(w)

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
        """
        Runs the hyper-parameter optimization program.

        :return:
        """
        self.init_search()

        # Configure the job
        bohb = BOHB(
            configspace=self.workers[0].get_configspace(self.config),
            run_id=self.config.get("bohb_search.run_id"),
            nameserver=self.config.get("bohb_search.host"),
            min_budget=self.config.get("bohb_search.min_budget"),
            max_budget=self.config.get("bohb_search.max_budget")
        )
        # Run it
        res = bohb.run(n_iterations=self.config.get("bohb_search.n_iter"),
                       min_n_workers=self.config.get("bohb_search.n_workers"))

        # Shut the job down
        bohb.shutdown(shutdown_workers=True)
        self.name_server.shutdown()

        # Print the best result
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()

        print('Best found configuration:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                    sum([r.budget for r in res.get_all_runs()]) / self.config.get("bohb_search.max_budget")))


class BOHBWorker(Worker):
    """
    Class of a worker for the BOHB hyper-parameter optimization algorithm.
    """

    def __init__(self, *args, sleep_interval=0, **kwargs):
        self.job_config = kwargs.pop('job_config')
        self.parent_job = kwargs.pop('parent_job')
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval
        self.next_trial_no = 0

    def compute(self, config, budget, **kwargs):
        """
        Creates a trial of the hyper-parameter optimization job and returns the best configuration of the trial.

        :param config: dictionary containing the sampled configurations by the optimizer
        :param budget: (float) amount of time/epochs/etc. the model can use to train
        :param kwargs:
        :return: dictionary containing the best hyper-parameter configuration of the trial
        """
        trial_no = self.next_trial_no

        parameters = config

        # create job for trial
        folder = str("{:05d}".format(trial_no))
        conf = self.job_config.clone(folder)
        conf.set("job.type", "train")
        conf.set_all(parameters)
        conf.init_folder()

        # run trial
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
        """
        Reads the config file and produces the necessary variables. Returns a configuration space with
        all variables and their definition.

        :param config: dictionary containing the variables and their possible values
        :return: ConfigurationSpace containing all variables.
        """
        config_space = CS.ConfigurationSpace()

        parameters = config.get("bohb_search.parameters")
        for p in parameters:
            v_name = p['name']
            v_type = p['type']

            if v_type == 'choice':
                config_space.add_hyperparameter(CSH.CategoricalHyperparameter(
                    v_name,
                    choices=p['values']
                ))
            elif v_type == 'range':
                log_scale = False
                if "log_scale" in p.keys():
                    log_scale = p['log_scale']
                config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(
                    name=v_name,
                    lower=p['bounds'][0],
                    upper=p['bounds'][1],
                    default_value=p['bounds'][1],
                    log=log_scale
                ))
            elif v_type == 'fixed':
                config_space.add_hyperparameter(CSH.Constant(
                    name=v_name,
                    value=p['value']
                ))
            else:
                raise ValueError("Unknown variable type")
        return config_space
