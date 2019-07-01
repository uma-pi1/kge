import torch


class KgeLoss:
    """ Wraps torch loss functions """
    def __init__(self):
        self._loss = None

    @staticmethod
    def create(config):
        """ Factory method for loss creation """
        if config.get("train.loss") == "ce":
            return torch.nn.CrossEntropyLoss(reduction="mean")
        elif config.get("train.loss") == "bce":
            return BceKgeLoss(reduction="mean", pos_weight=None)
        elif config.get("train.loss") == "margin_ranking":
            # TODO decide where to put margin value in config
            margin = config.get("margin_ranking.margin")
            return MarginRankingKgeLoss(margin,
                                        config,
                                        reduction="mean")
        elif config.get("train.loss") == "kl":
            return torch.nn.KLDivLoss(reduction="mean")
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.loss")

    def __call__(self, scores, labels=None):
        return self._compute_loss(scores, labels)

    def _compute_loss(self, scores, labels):
        raise NotImplementedError()


class BceKgeLoss(KgeLoss):
    def __init__(self, reduction="mean", pos_weight=None):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight
        )

    def _compute_loss(self, scores, labels):
        return self._loss(scores.view(-1), labels.view(-1))


class MarginRankingKgeLoss(KgeLoss):
    def __init__(self, margin, config, reduction="mean"):
        super().__init__()
        self._device = config.get("job.device")
        self._training_type = config.get("train.type")
        self._num_negatives = config.get("negative_sampling.num_negatives_s")
        self._loss = torch.nn.MarginRankingLoss(
            margin=margin, reduction=reduction
        )

    def _compute_loss(self, scores, labels):
        if self._training_type == "negative_sampling":
            # scores and labels are (batch_size * num_negatives, 1)
            # Pair each 1 with the following zeros until a 1 appears
            pos_positives = labels.nonzero().to(self._device).view(-1)
            pos_negatives = (labels == 0).nonzero().to(self._device).view(-1)
            pos_positives = pos_positives.repeat(1, self._num_negatives).view(-1)
            positives = scores[pos_positives]
            negatives = scores[pos_negatives]
            target = torch.ones(positives.size())

            return self._loss(positives, negatives, target)
        else:
            # TODO determine how to form pairs for margin ranking in 1toN training
            # scores and labels are tensors of size (batch_size, num_entities)
            # Each row has 1s and 0s of a single sp or po tuple from training
            # How to combine them for pairs?
            # Each 1 with all 0s?
            raise NotImplementedError("Margin ranking with 1toN training not yet supported.")
