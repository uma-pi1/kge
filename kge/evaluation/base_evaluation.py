from typing import List


class BaseEvaluation:
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

		raise NotImplementedError
