from __future__ import annotations

from kge import Config, Dataset
from kge.model import KgeModel
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
    def create(config: Config, dataset: Dataset, parent_job=None, model=None):
        from kge.job import TrainingJob, EvaluationJob, SearchJob

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
    def resume(
        checkpoint: str = None,
        config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
    ) -> Job:
        if checkpoint is None and config is None:
            raise ValueError(
                "Please provide either the config file located in the folder structure "
                "containing the checkpoint or the checkpoint itself."
            )
        elif checkpoint is None:
            last_checkpoint = config.last_checkpoint()
            if last_checkpoint is not None:
                checkpoint = config.checkpoint_file(last_checkpoint)

        if checkpoint is not None:
            job = Job.load_from(checkpoint, config, dataset, parent_job=parent_job)
            if type(checkpoint) == str:
                job.config.log("Loading checkpoint from {}...".format(checkpoint))
        else:
            job = Job.create(config, dataset, parent_job=parent_job)
            job.config.log("No checkpoint found, starting from scratch...")
        return job

    @classmethod
    def load_from(
        cls,
        checkpoint: Union[str, Dict],
        config: Config = None,
        dataset: Dataset = None,
        parent_job=None,
    ) -> Job:
        if config is not None:
            device = config.get("job.device")
        else:
            device = "cpu"
        if type(checkpoint) == str:
            checkpoint = load_checkpoint(checkpoint, device)
        config = Config.load_from(
            checkpoint["config"], config, folder=checkpoint["folder"]
        )
        model: KgeModel = None
        dataset = None
        if checkpoint["model"] is not None:
            model = KgeModel.load_from(checkpoint, config=config, dataset=dataset)
            dataset = model.dataset
        if dataset is None:
            dataset = Dataset.load(config)
        job = Job.create(config, dataset, parent_job, model)
        job.load(checkpoint, model)
        return job

    def load(self, checkpoint, model):
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
