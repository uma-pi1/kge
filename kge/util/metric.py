from kge import Config
from kge.job import Job
from typing import Union, Iterable, List


class Metric:
    "Utility class for comparing metrics."

    def __init__(self, metric_max: Union[Job, Config, bool]):
        "Params: metric_max=True means higher is better."
        if isinstance(metric_max, Job):
            metric_max = metric_max.config
        if isinstance(metric_max, Config):
            metric_max = metric_max.get("valid.metric_max")
        self._metric_max = metric_max

    def better(self, metric1: float, metric2: float) -> bool:
        if self._metric_max:
            return metric1 > metric2
        else:
            return metric1 < metric2

    def best(self, metrics: Iterable[float]) -> float:
        if self._metric_max:
            return max(metrics)
        else:
            return min(metrics)

    def best_index(self, metrics: List[float]) -> int:
        return metrics.index(self.best(metrics))

    def worst(self) -> float:
        if self._metric_max:
            return float("-Inf")
        else:
            return float("Inf")
