from kge import Config, Dataset

from typing import Optional


class JobFactory:
    @staticmethod
    def pick_job(job_type=None, config=None):
        from kge.job import TrainingJob, EvaluationJob, SearchJob

        if job_type is None and config is None:
            job_type = "train"
        elif job_type is None:
            job_type = config.get("job.type")

        if job_type == "train":
            return TrainingJob
        elif job_type == "search":
            return SearchJob
        elif job_type == "eval":
            return EvaluationJob
        else:
            raise ValueError("unknown job type")

    @staticmethod
    def create(config: Config, dataset: Dataset, parent_job=None, init=True):
        job_class = JobFactory.pick_job(config=config)
        return job_class.create(config, dataset, parent_job=parent_job, init=init)

    @staticmethod
    def load_from(
        checkpoint_file: str,
        job_type=None,
        config: Optional[Config] = None,
        dataset: Optional[Dataset] = None,
        parent_job=None,
    ):
        job_class = JobFactory.pick_job(job_type=job_type, config=config)
        return job_class.load_from(
            checkpoint_file, config=config, dataset=dataset, parent_job=parent_job
        )
