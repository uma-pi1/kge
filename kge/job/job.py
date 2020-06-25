from __future__ import annotations

from kge import Config, Dataset
from kge.util import load_checkpoint
import uuid

from kge.misc import get_git_revision_short_hash
import os
import socket
from typing import Any, Callable, Dict, List, Optional, Union


def _trace_job_creation(job: "Job"):
    """Create a trace entry for a job"""
    from torch import __version__ as torch_version

    userhome = os.path.expanduser("~")
    username = os.path.split(userhome)[-1]
    job.trace_entry = job.trace(
        git_head=get_git_revision_short_hash(),
        torch_version=torch_version,
        username=username,
        hostname=socket.gethostname(),
        folder=job.config.folder,
        event="job_created",
    )


def _save_job_config(job: "Job"):
    """Save the job configuration"""
    config_folder = os.path.join(job.config.folder, "config")
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)
    job.config.save(os.path.join(config_folder, "{}.yaml".format(job.job_id[0:8])))


class Job:
    # Hooks run after job creation has finished
    # signature: job
    job_created_hooks: List[Callable[["Job"], Any]] = [
        _trace_job_creation,
        _save_job_config,
    ]

    def __init__(self, config: Config, dataset: Dataset, parent_job: "Job" = None):
        self.config = config
        self.dataset = dataset
        self.job_id = str(uuid.uuid4())
        self.parent_job = parent_job
        self.resumed_from_job_id: Optional[str] = None
        self.trace_entry: Dict[str, Any] = {}
        self._is_prepared = False

        # prepend log entries with the job id. Since we use random job IDs but
        # want short log entries, we only output the first 8 bytes here
        self.config.log_prefix = "[" + self.job_id[0:8] + "] "

        if self.__class__ == Job:
            for f in Job.job_created_hooks:
                f(self)

        #: Hooks before running a job
        #: Signature: job
        self.pre_run_hooks: List[Callable[[Job], Any]] = []

        #: Hooks after running a job
        #: Signature: job, dict returned by the run method
        self.post_run_hooks: List[Callable[[Job, Dict], Any]] = []


    @staticmethod
    def create(
        config: Config, dataset: Optional[Dataset] = None, parent_job=None, model=None
    ):
        "Create a new job."
        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if dataset is None:
            dataset = Dataset.create(config)

        job_type = config.get("job.type")
        if job_type == "train":
            return TrainingJob.create(
                config, dataset, parent_job=parent_job, model=model
            )
        elif job_type == "search":
            return SearchJob.create(config, dataset, parent_job=parent_job)
        elif job_type == "eval":
            return EvaluationJob.create(
                config, dataset, parent_job=parent_job, model=model
            )
        else:
            raise ValueError("unknown job type")

    @classmethod
    def create_from(
        cls,
        checkpoint: Dict,
        new_config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
    ) -> Job:
        """
        Creates a Job based on a checkpoint
        Args:
            checkpoint: loaded checkpoint
            new_config: optional config object - overwrites options of config
                              stored in checkpoint
            dataset: dataset object
            parent_job: parent job (e.g. search job)

        Returns: Job based on checkpoint

        """
        from kge.model import KgeModel

        model: KgeModel = None
        # search jobs don't have a model
        if "model" in checkpoint and checkpoint["model"] is not None:
            model = KgeModel.create_from(
                checkpoint, new_config=new_config, dataset=dataset
            )
            config = model.config
            dataset = model.dataset
        else:
            config = Config.create_from(checkpoint)
            if new_config:
                config.load_config(new_config)
            dataset = Dataset.create_from(checkpoint, config, dataset)
        job = Job.create(config, dataset, parent_job, model)
        job._load(checkpoint)
        job.config.log("Loaded checkpoint from {}...".format(checkpoint["file"]))
        return job

    def _load(self, checkpoint: Dict):
        """Job type specific operations when created from checkpoint.

        Called during `create_from`. Assumes that config, dataset, and model have
        already been loaded from the specified checkpoint.

        """
        pass

    def _prepare(self):
        pass

    def run(self):
        """
        Run the job: first prepare it run some pre run hooks, then execute the job
        and run some post run hooks and return the result.
        :return: Output of the job, if any.
        """
        if not self._is_prepared:
            self._prepare()

        for f in self.pre_run_hooks:
            f(self)

        result = self._run()

        for f in self.post_run_hooks:
            f(self, result)

        return result

    def _run(self):
        raise NotImplementedError

    def trace(self, **kwargs) -> Dict[str, Any]:
        """Write a set of key-value pairs to the trace file and automatically append
        information about this job. See `Config.trace` for more information."""
        if self.parent_job is not None:
            kwargs["parent_job_id"] = self.parent_job.job_id
        if self.resumed_from_job_id is not None:
            kwargs["resumed_from_job_id"] = self.resumed_from_job_id

        return self.config.trace(
            job_id=self.job_id, job=self.config.get("job.type"), **kwargs
        )
