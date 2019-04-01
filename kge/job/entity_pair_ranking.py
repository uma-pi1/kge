from kge.job import EvalJob


class EntityPairRanking(EvalJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, model):
        super().__init__(config, dataset, model)
