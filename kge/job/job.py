from kge import Config, Dataset
import uuid

from kge.misc import get_git_revision_short_hash
import os
import socket
from typing import Any, Callable, Dict, List, Optional


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

    def resume(self, checkpoint_file: str = None):
        """Load job state from last or specified checkpoint.

        Restores all relevant state to resume a previous job. To run the restored job,
        use :func:`run`.

        Should set `resumed_from_job` to the job ID of the previous job.

        """
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def create(config: Config, dataset: Dataset, parent_job: "Job" = None) -> "Job":
        """Creates a job for a given configuration."""

        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if config.get("job.type") == "train":
            job = TrainingJob.create(config, dataset, parent_job)
        elif config.get("job.type") == "search":
            job = SearchJob.create(config, dataset, parent_job)
        elif config.get("job.type") == "eval":
            job = EvaluationJob.create(config, dataset, parent_job)
        else:
            raise ValueError("unknown job type")

        return job

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
