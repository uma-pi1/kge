from typing import List


class BaseEvaluator:
    """
    Interface
    """

    def compute_metrics(self, predictions, labels, filters) -> List[float]:
        """

        :param predictions:
        :param labels:
        :param filters:
        :return: list of metrics (floats)
        """

        raise NotImplemented


class OneToNEvaluator(BaseEvaluator):
    """ Entity ranking protocol """


class NToNEvaluator(BaseEvaluator):
    """ Entity-pair ranking protocol """
