class Job:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def resume(self):
        """Restores all relevant state to resume a previous job.

        To run the restored job, use :func:`run`.

        """
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def create(config, dataset):
        """Creates a job for a given configuration."""

        from kge.job import TrainingJob, GridJob

        if config.get('job.type') == 'train':
            return TrainingJob.create(config, dataset)
        elif config.get('job.type') == 'grid':
            return GridJob(config, dataset)
        else:
            raise ValueError("unknown job type")
