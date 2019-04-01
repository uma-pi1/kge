from kge.job import Job


class EvaluationJob(Job):

    def __init__(self, config, dataset, model=None):

        from kge.job import TrainingJob

        self.config = config
        self.dataset = dataset
        self.batch_size = config.get('train.batch_size')
        self.device = self.config.get('job.device')
        self.k = self.config.get('eval.k')
        if model:
            self.model = model
        else:
            training_job = TrainingJob.create(config, dataset)
            training_job.resume()
            self.model = training_job.model

    def create(config, dataset, model=None):
        """Factory method to create an evaluation job """

        from kge.job import EntityRanking, EntityPairRanking

        if config.get('eval.type') == 'entity_ranking':
            return EntityRanking(config, dataset, model)
        elif config.get('eval.type') == 'entity_pair_ranking':
            return EntityPairRanking(config, dataset, model)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("eval.type")

    def run(self):
        """ Compute evaluation metrics, output results to trace file """
        raise NotImplementedError

    # TODO needs resume method because of inheritance, used for what here?
    # -> mark as invalid
