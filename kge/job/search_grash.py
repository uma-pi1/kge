import concurrent.futures
import logging
import torch.cuda
import math
import random
import kge.job.search
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import torch.multiprocessing as mp
import os
import sys
import shutil
import gc

from kge.job import AutoSearchJob
from kge import Config, Dataset
from kge.util.package import package_model
from kge.config import _process_deprecated_options
from hpbandster.optimizers import HyperBand
from hpbandster.core.worker import Worker
from argparse import Namespace
from collections import defaultdict
from multiprocessing import Manager
from kge.util import get_configspace


class GraSH(HyperBand):
    """
    Our implementation of GraSH builds upon the HyperBand class of HPBandster
    (https://github.com/automl/HpBandSter). We extend it by distributing work
    among free devices in this class.
    """

    def __init__(
        self,
        free_devices,
        trial_dict,
        id_dict,
        configspace=None,
        eta=3,
        min_budget=0.01,
        max_budget=1,
        **kwargs,
    ):
        self.free_devices = free_devices
        self.trial_dict = trial_dict
        self.id_dict = id_dict
        self.assigned_devices = defaultdict(lambda: None)
        super(GraSH, self).__init__(configspace, eta, min_budget, max_budget, **kwargs)

    def _submit_job(self, config_id, config, budget):
        # This method is called before computation of a trial. We use it
        # to assign a free device.
        with self.thread_cond:
            self.assigned_devices[config_id] = self.free_devices.pop()
            config["job.device"] = self.assigned_devices[config_id]

        return super(GraSH, self)._submit_job(config_id, config, budget)

    def job_callback(self, job):
        # Free the device once the job is finished.
        with self.thread_cond:
            self.free_devices.append(self.assigned_devices[job.id])
            del self.assigned_devices[job.id]
        return super(GraSH, self).job_callback(job)


class GraSHSearchJob(AutoSearchJob):
    """
    Job for hyperparameter search using GraSH (Kochsiek et al. 2022)
    Source: https://github.com/uma-pi1/grash
    """

    def __init__(self, config: Config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.name_server = None  # Server address to run the job on
        self.workers = []  # Workers that will run in parallel
        manager = Manager()
        # Create dict for prior result storage
        self.trial_dict = manager.dict()
        # Create empty dict for id generation
        self.id_dict = manager.dict()
        self.processes = []
        self.sh_rounds = 0
        self.eta = 0
        self.num_trials = 0
        self.total_computations = 0
        self.subset_stats = dict()
        self.subsets = dict()
        # self.dataset = dataset
        self.original_dataset = self.dataset
        self.k_core_manager = None

    def init_search(self):

        # perform plausibility checks for GraSH parameter settings
        self.check_grash_parameters()

        # generate random seed if it is <0
        if self.config.get("grash_search.seed") < 0:
            # create new instance of random to make sure that no seed is set for randint
            grash_random = random.Random()
            seed = grash_random.randint(0, 100)
            self.config.set("grash_search.seed", seed)
            self.config.save(os.path.join(self.config.folder, "config.yaml"))

        # Start an HPBandster nameserver to organize multiple workers
        run_id = os.path.basename(self.config.folder)
        self.name_server = hpns.NameServer(host=None, port=None, run_id=run_id)
        self.name_server.start()

        # Load prior results if available
        if self.results:
            for k, v in self.results[0].items():
                self.trial_dict[k] = v

        # Determine corresponding Hyperband configuration for GraSH configuration
        self.num_trials = self.config.get("grash_search.num_trials")
        self.eta = self.config.get("grash_search.eta")
        sh_rounds = math.log(self.num_trials, self.eta)
        if not sh_rounds.is_integer():
            if self.config.get("job.auto_correct"):
                sh_rounds = math.floor(sh_rounds)
                self.num_trials = self.eta**sh_rounds
                self.config.log(
                    "Setting grash_search.num_trials to {}, was set to {} and needs to "
                    "equal a positive integer power of eta.".format(
                        self.num_trials, self.config.get("grash_search.num_trials")
                    )
                )
            else:
                raise Exception(
                    "grash_search.num_trials was set to {}, "
                    "needs to equal a positive integer power of eta.".format(
                        self.num_trials
                    )
                )
        self.sh_rounds = int(sh_rounds)

        # Calculate the total number of computations
        for i in range(self.sh_rounds + 1):
            self.total_computations += int(self.num_trials / (self.eta**i))

        # Perform k-core decomposition if not done yet
        if self.config.get("grash_search.variant") != "epoch":
            # add full dataset to subset dict
            self.subsets[0] = self.original_dataset
            # get k_core_stats
            # self.subset_stats = self.dataset.index("k_core_stats")
            self.k_core_manager = self.dataset.index("k-cores")
            self.subset_stats = self.k_core_manager.get_k_core_stats()

        # Create workers (HPBandSter logging is currently not shown to the user)
        worker_logger = logging.getLogger()
        for i in range(self.config.get("search.num_workers")):
            w = GraSHWorker(
                nameserver=None,
                logger=worker_logger,
                run_id=run_id,
                job_config=self.config,
                parent_job=self,
                trial_dict=self.trial_dict,
                id_dict=self.id_dict,
                id=i,
            )

            # w.run(background=True)

            p = mp.Process(target=w.run, args=(False,))
            self.processes.append(p)
            p.start()

            self.workers.append(w)

    def register_trial(self, parameters=None):
        # HPBandster does this itself
        pass

    def register_trial_result(self, trial_id, parameters, trace_entry):
        # HPBandster does this itself
        pass

    def get_best_parameters(self):
        # HPBandster does this itself
        pass

    def run(self):
        """
        Runs the hyper-parameter optimization program.
        :return:
        """
        self.init_search()
        result_logger = hpres.json_result_logger(
            directory=self.config.folder, overwrite=True
        )

        # get configspace object corresponding to given LibKGE config
        hyperparameter_space = self.config.get("grash_search.parameters")
        configspace_seed = self.config.get("grash_search.seed")
        configspace = get_configspace(hyperparameter_space, configspace_seed)

        # Configure the job
        grash = GraSH(
            free_devices=self.free_devices,
            configspace=configspace,
            run_id=os.path.basename(self.config.folder),
            nameserver=None,
            result_logger=result_logger,
            eta=self.eta,
            min_budget=1 / self.num_trials,
            max_budget=1,
            trial_dict=self.trial_dict,
            id_dict=self.id_dict,
        )

        self.config.log("Started the GraSH hyperparameter search")

        # calculation for future Hyperband extension (as defined in Hyperband paper)
        # hpb_iterations = math.floor(math.log(num_trials, eta)) + 1

        # Run GraSH by setting number of Hyperband iterations to 1
        hpb_iterations = 1
        grash.run(
            n_iterations=hpb_iterations,
            min_n_workers=self.config.get("search.num_workers"),
        )

        # Shut it down
        grash.shutdown(shutdown_workers=True)
        self.name_server.shutdown()
        for p in self.processes:
            p.terminate()

    def check_grash_parameters(self):
        """
        Perform plausibility checks for GraSH parameter settings.
        """
        if self.config.get("grash_search.variant") not in [
            "combined",
            "epoch",
            "graph"
        ]:
            raise ValueError(
                f"GraSH variant"
                f"{self.config.get('grash_search.variant')} is not supported."
            )
        if self.config.get("grash_search.cost_metric") not in [
            "triples_and_entities",
            "triples",
        ]:
            raise ValueError(
                f"GraSH cost metric"
                f"{self.config.get('grash_search.cost_metric')} is not supported."
            )
        if self.config.get("grash_search.eta") <= 1:
            raise ValueError(
                f"GraSH eta"
                f"{self.config.get('grash_search.eta')} not supported. Must be > 1."
            )
        if self.config.get("grash_search.num_trials") <= 1:
            raise ValueError(
                f"GraSH num_trials {self.config.get('grash_search.num_trials')}"
                f"not supported. Must be > 1."
            )
        if self.config.get("grash_search.search_budget") <= 0:
            raise ValueError(
                f"GraSH search_budget {self.config.get('grash_search.search_budget')}"
                f"not supported. Must be > 0."
            )
        if self.config.get("grash_search.valid_frac") >= 1:
            raise ValueError(
                f"GraSH valid_frac {self.config.get('grash_search.valid_frac')}"
                f"not supported. Must be < 1."
            )
        if self.config.get("grash_search.min_negatives_percentage") > 1:
            raise ValueError(
                f"GraSH min_negatives_percentage"
                f"{self.config.get('grash_search.min_negatives_percentage')}"
                f"not supported. Must be <= 1."
            )


class GraSHWorker(Worker):
    """
    Class of a worker for the GraSH hyperparameter optimization algorithm. It is
    responsible for creating LibKGE train jobs for the hyperparameter configurations
    that are passed from the GraSH master. By creating multiple workers, we achieve
    parallelization of trial execution.
    """

    def __init__(self, *args, **kwargs):
        self.job_config = kwargs.pop("job_config")
        self.parent_job = kwargs.pop("parent_job")
        self.trial_dict = kwargs.pop("trial_dict")
        self.id_dict = kwargs.pop("id_dict")
        self.search_worker_id = kwargs.get("id")
        super().__init__(*args, **kwargs)

    def compute(self, config_id, config, budget, **kwargs):
        try:
            return self._compute(config_id, config, budget)
        except Exception as e:
            self.parent_job.config.log(f"Aborting GraSH search due to failure: {e}")
            os._exit(1)

    def _compute(self, config_id, config, budget):
        """
        Creates a LibKGE train job for a given hyperparameter configuration and returns
        its result to the GraSH master. This includes setting the subgraph and training
        epochs corresponding to a given trial budget.
        :param config_id: a triplet of ints that uniquely identifies a configuration.
        The convention is id = (iteration, budget index, running index)
        :param config: dictionary containing the sampled configurations by the optimizer
        :param budget: (float) relative budget available for this trial in the current
        round
        :return: the loss (1-score) of this trial
        """

        # extract the Hyperband iteration and config number from the config_id
        hpb_iter = config_id[0]
        config_no = config_id[2]

        # the successive halving iteration is not provided by HPBandster, so we keep
        # track of it ourself
        if (hpb_iter, config_no) in self.id_dict:
            self.id_dict[(hpb_iter, config_no)] += 1
        else:
            self.id_dict[(hpb_iter, config_no)] = 0
        sh_iter = self.id_dict[(hpb_iter, config_no)]

        # get the trial ID
        trial_id = self.get_trial_id(hpb_iter, sh_iter, config_no)

        # determine the job number (-1 because it will be increased by 1 within the
        # train job)
        job_no = sum(self.id_dict.values()) + len(self.id_dict.keys()) - 1

        # create job for trial and set the hyperparameters
        conf = self.job_config.clone(trial_id)
        conf.set("job.type", "train")
        conf.set_all(_process_deprecated_options(config))

        # check if trial result is already available for the given hyperparameters
        if trial_id in self.trial_dict:
            set_new = set(config.items())
            set_old = set(self.trial_dict[trial_id][1].items())
            difference = set_old - set_new
            if not difference:
                valid_metric = conf.get("valid.metric")
                best_score = self.trial_dict[trial_id][0]
                conf.log(
                    f"Trial {conf.folder} registered with {valid_metric} {best_score}"
                )
                return {"loss": 1 - best_score, "info": {}}
            else:
                # raise an exception if HPBandster generates different hyperparameter
                # trials because the random seed has changed
                raise Exception(
                    "The seed for generating hyperparameter trials with GraSH has "
                    "changed. To resume the execution of GraSH, please set the old seed"
                    " under grash_search.seed in the config file."
                )

        # scale given budget based on the search budget in terms of full train runs on
        # the whole graph
        if sh_iter < self.parent_job.sh_rounds:
            budget = budget * (
                self.parent_job.config.get("grash_search.search_budget")
                / self.parent_job.sh_rounds
            )

        # in the combined variant, we share savings equally between graph and epochs by
        # taking the square root. This allows to use more epochs and larger subgraphs.
        if self.parent_job.config.get("grash_search.variant") == "combined":
            budget = math.sqrt(budget)

        # determine the epochs for this trial and set them
        epochs = self._determine_epochs(self.parent_job.config, budget)
        conf.set("train.max_epochs", epochs)

        # determine the subset for this trial
        if self.parent_job.config.get("grash_search.variant") != "epoch":
            # determine the largest subset that has cost <= budget
            subset = next(
                (
                    x
                    for x in self.parent_job.subset_stats.items()
                    if x[1][self.parent_job.config.get("grash_search.cost_metric")]
                    <= budget
                ),
                None,
            )
            if subset is None:
                raise ValueError(f"no fitting subgraph for size_budget {budget} found")

            # get subset from original dataset and set it as dataset for the parent job
            self.parent_job.dataset = self.parent_job.k_core_manager.get_k_core_dataset(subset[0])
            conf.set("dataset", self.parent_job.dataset.config.get("dataset"))

            # downscale number of negatives (don't if slot has default option -1)
            number_samples_s = conf.get("negative_sampling.num_samples.s")
            if number_samples_s > 0:
                negatives_scaler = max(
                    subset[1]["rel_entities"],
                    self.parent_job.config.get("grash_search.min_negatives_percentage"),
                )
                conf.set(
                    "negative_sampling.num_samples.s",
                    math.ceil(number_samples_s * negatives_scaler),
                )
            number_samples_o = conf.get("negative_sampling.num_samples.o")
            if number_samples_o > 0:
                conf.set(
                    "negative_sampling.num_samples.o",
                    math.ceil(number_samples_o * negatives_scaler),
                )

            # reuse the predecessor model checkpoint if available to keep initialization
            if sh_iter != 0:
                predecessor_trial_id = self.get_trial_id(
                    hpb_iter, (sh_iter - 1), config_no
                )
                path_to_model = ""
                if conf.get("grash_search.keep_initialization"):
                    path_to_model = os.path.join(
                        f"{os.path.dirname(conf.folder)}",
                        f"{predecessor_trial_id}",
                        f"model_00000.pt",
                    )
                if conf.get("grash_search.keep_pretrained"):
                    path_to_model = os.path.join(
                        f"{os.path.dirname(conf.folder)}",
                        f"{predecessor_trial_id}",
                        f"model_best.pt",
                    )
                conf.set("lookup_embedder.pretrain.model_filename", path_to_model)

        # save config.yaml
        conf.init_folder()

        # copy the last checkpoint from the previous round to the new folder in the
        # epoch variant to save resources by continuing training
        if (
            self.parent_job.config.get("grash_search.variant") == "epoch"
            and sh_iter != 0
        ):
            predecessor_trial_id = self.get_trial_id(hpb_iter, (sh_iter - 1), config_no)
            config_pred = Config(
                folder=os.path.join(os.path.dirname(conf.folder), predecessor_trial_id),
                load_default=False,
            )
            last_checkpoint = config_pred.last_checkpoint_number()
            if last_checkpoint:
                filename = f"checkpoint_{str('{:05d}'.format(last_checkpoint))}.pt"
                shutil.copy(
                    os.path.join(
                        os.path.dirname(conf.folder), predecessor_trial_id, filename
                    ),
                    os.path.join(conf.folder, filename),
                )
                conf.log(
                    f"Copied the predecessor checkpoint for trial {trial_id}. "
                    f"Resuming from there."
                )
            else:
                conf.log(
                    f"Could not copy predecessor checkpoint for trial {trial_id}. "
                    f"Starting new round from scratch."
                )

        # run trial
        best = kge.job.search._run_train_job(
            (
                self.parent_job,
                job_no,
                conf,
                self.parent_job.total_computations,
                list(config.keys()),
            )
        )

        # save package checkpoint
        args = Namespace()
        args.checkpoint = None
        if conf.get("grash_search.keep_initialization"):
            args.checkpoint = f"{conf.folder}/checkpoint_00000.pt"
        if conf.get("grash_search.keep_pretrained"):
            args.checkpoint = f"{conf.folder}/checkpoint_best.pt"
        if args.checkpoint is not None:
            args.file = None
            package_model(args, self.parent_job.dataset)

        best_score = best[1]["metric_value"]
        del best

        # kill all active cuda tensors
        gc.collect()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == "cuda":
                    # yield obj
                    del obj
            except:
                pass
        with torch.cuda.device(conf.get("job.device")):
            torch.cuda.empty_cache()
        gc.collect()

        # remove device from hyperparameter dict
        config.pop("job.device", None)
        # add score and hyperparameters to trial dict
        self.trial_dict[trial_id] = [best_score, config]

        # save search checkpoint - trial_dict with results is synchronized due to the
        # use of Manager.dict()
        filename = f"{os.path.dirname(conf.folder)}//checkpoint_00001.pt"
        self.parent_job.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "type": "search_grash",
                "parameters": [],
                "results": [self.trial_dict._getvalue()],
                "job_id": self.parent_job.job_id,
            },
            filename,
        )

        return {"loss": 1 - best_score, "info": {"metric_value": best_score}}

    def _determine_epochs(self, config, budget):
        """
        Determines the training epochs for a given LigKGE configuration and trial budget
        """

        # differentiate between the GraSH variants
        if self.parent_job.config.get("grash_search.variant") == "graph":
            # in the graph variant, we do not downscale the epochs
            epochs = self.parent_job.config.get("train.max_epochs")
        else:
            epochs = math.floor(self.parent_job.config.get("train.max_epochs") * budget)
            if epochs < 1:
                raise ValueError(
                    f"Training epochs for budget {budget} is below 1 epoch and not"
                    f"supported, please increase train.max_epochs."
                )

        return epochs

    @staticmethod
    def get_trial_id(hpb_iter, sh_iter, config_no):
        """
        Generates a trial ID for a given hpb iteration, sh iteration, and config number.
        The trial ID serves as folder name and has 8 digits:
        <2 digits for hpb_iter><2 digits for sh_iter><4 digits for config_no>
        """
        trial_id = (
            str("{:02d}".format(hpb_iter))
            + str("{:02d}".format(sh_iter))
            + str("{:04d}".format(config_no))
        )

        return trial_id
