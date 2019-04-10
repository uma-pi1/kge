from kge.job import EvaluationJob


class EntityPairRankingJob(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
