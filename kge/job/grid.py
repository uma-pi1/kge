from kge.job import Job


class GridJob(Job):
    """Job to perform grid search."""
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
