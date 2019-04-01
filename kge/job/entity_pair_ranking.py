from kge.job import EvaluationJob


class EntityPairRanking(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
