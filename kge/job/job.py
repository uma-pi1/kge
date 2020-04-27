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

        # prepend log entries with the job id. Since we use random job IDs but
        # want short log entries, we only output the first 8 bytes here
        self.config.log_prefix = "[" + self.job_id[0:8] + "] "

        if self.__class__ == Job:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config, dataset: Optional[Dataset] = None, parent_job=None, model=None
    ):
        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if dataset is None:
            dataset = Dataset.load(dataset)

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

    @staticmethod
    def find_and_create_from(
        checkpoint: str = None,
        config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
    ) -> Job:
        """
        Finds and loads the checkpoint to resume from given a config file in in the
        experiment folder structure.
        If a checkpoint is specified, it will be loaded. In this case no config file
        needs to be specified.
        Args:
            checkpoint: path to checkpoint file
            config: config in the experiment folder structure
            dataset: dataset object
            parent_job: parent job (e.g. search job)

        Returns: Job based on checkpoint or new Job if no checkpoint found

        """
        if checkpoint is None and config is None:
            raise ValueError(
                "Config or checkpoint required."
            )
        if checkpoint is None:
            last_checkpoint = config.last_checkpoint()
            if last_checkpoint is not None:
                checkpoint = config.checkpoint_file(last_checkpoint)

        if checkpoint is not None:
            job = Job.create_from(checkpoint, config, dataset, parent_job=parent_job)
            if type(checkpoint) == str:
                job.config.log("Loading checkpoint from {}...".format(checkpoint))
            else:
                job.config.log("Loaded checkpoint.")
        else:
            job = Job.create(config, dataset, parent_job=parent_job)
            job.config.log("No checkpoint found or specified, starting from scratch...")
        return job

    @classmethod
    def create_from(
        cls,
        checkpoint: Union[str, Dict],
        overwrite_config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
    ) -> Job:
        """
        Creates a Job based on a checkpoint
        Args:
            checkpoint: path to checkpoint file or loaded checkpoint
            overwrite_config: optional config object - overwrites options of config
                              stored in checkpoint
            dataset: dataset object
            parent_job: parent job (e.g. search job)

        Returns: Job based on checkpoint

        """
        from kge.model import KgeModel

        if overwrite_config is not None and overwrite_config.exists("job.device"):
            device = overwrite_config.get("job.device")
        else:
            device = "cpu"
        if type(checkpoint) == str:
            checkpoint = load_checkpoint(checkpoint, device)
        overwrite_config = Config.create_from(checkpoint, overwrite_config)
        model: KgeModel = None
        # search jobs don't have a model
        if "model" in checkpoint and checkpoint["model"] is not None:
            model = KgeModel.create_from(
                checkpoint, config=overwrite_config, dataset=dataset
            )
            dataset = model.dataset
        else:
            dataset = Dataset.create_from(checkpoint, overwrite_config, dataset)
        job = Job.create(overwrite_config, dataset, parent_job, model)
        job.load(checkpoint, model)
        return job

    def load(self, checkpoint: Dict, model):
        """Job type specific operations when loaded from checkpoint"""
        pass

    def run(self):
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
