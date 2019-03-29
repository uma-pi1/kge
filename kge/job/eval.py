from kge.job import Job


class EvalJob(Job):

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.batch_size = config.get('train.batch_size')
        self.device = self.config.get('job.device')

    def create(config, dataset):
        """Factory method to create an evaluation job """

        from kge.job import EntityRanking, EntityPairRanking

        if config.get('evaluation.type') == 'entity_ranking':
            return EntityRanking(config, dataset)
        elif config.get('evaluation.type') == 'entity_pair_ranking':
            return EntityPairRanking(config, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("evaluation.type")

    def run(self):
        """ Compute evaluation metrics, output results to trace file """
        pass

    # TODO needs resume method because of inheritance, used for what here?
