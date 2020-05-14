import math
import torch
import torch.nn.functional as F
from kge import Config

# Documented losses
# - See description in config-default.yaml
#
# Other, undocumented losses. EXPERIMENTAL, may be removed.
#
# bce_mean (not KvsAll): as BCE but for each positive triple, average the BCE of the
# positive triple and the *mean* BCE of its negative triples. Used in RotatE paper and
# implementation.
#
# bce_self_adversarial (not KvsAll): as bce_mean, but average the negative triples
# weighted by a softmax over their scores. Temperature is taken from
# "user.bce_self_adversarial_temperature" if specified there.
class KgeLoss:
    """A loss function.

    When applied to a batch, the resulting loss MUST NOT be averaged by the batch size.

    """

    def __init__(self, config: Config):
        self.config = config
        self._loss = None

    @staticmethod
    def create(config: Config):
        """Factory method for loss function instantiation."""

        # perhaps TODO: try class with specified name -> extensibility
        config.check(
            "train.loss",
            [
                "bce",
                "bce_mean",
                "bce_self_adversarial",
                "margin_ranking",
                "ce",
                "kl",
                "soft_margin",
                "se",
            ],
        )
        if config.get("train.loss") == "bce":
            offset = config.get("train.loss_arg")
            if math.isnan(offset):
                offset = 0.0
                config.set("train.loss_arg", offset, log=True)
            return BCEWithLogitsKgeLoss(config, offset=offset, bce_type=None)
        elif config.get("train.loss") == "bce_mean":
            offset = config.get("train.loss_arg")
            if math.isnan(offset):
                offset = 0.0
                config.set("train.loss_arg", offset, log=True)
            return BCEWithLogitsKgeLoss(config, offset=offset, bce_type="mean")
        elif config.get("train.loss") == "bce_self_adversarial":
            offset = config.get("train.loss_arg")
            if math.isnan(offset):
                offset = 0.0
                config.set("train.loss_arg", offset, log=True)
            try:
                temperature = float(config.get("user.bce_self_adversarial_temperature"))
            except KeyError:
                temperature = 1.0
            config.log(f"Using adversarial temperature {temperature}")
            return BCEWithLogitsKgeLoss(
                config,
                offset=offset,
                bce_type="self_adversarial",
                temperature=temperature,
            )
        elif config.get("train.loss") == "kl":
            return KLDivWithSoftmaxKgeLoss(config)
        elif config.get("train.loss") == "margin_ranking":
            margin = config.get("train.loss_arg")
            if math.isnan(margin):
                margin = 1.0
                config.set("train.loss_arg", margin, log=True)
            return MarginRankingKgeLoss(config, margin=margin)
        elif config.get("train.loss") == "soft_margin":
            return SoftMarginKgeLoss(config)
        elif config.get("train.loss") == "se":
            return SEKgeLoss(config)
        else:
            raise ValueError(
                "invalid value train.loss={}".format(config.get("train.loss"))
            )

    def __call__(self, scores, labels, **kwargs):
        """Computes the loss given the scores and corresponding labels.

        `scores` is a batch_size x triples matrix holding the scores predicted by some
        model.

        `labels` is either (i) a batch_size x triples Boolean matrix holding the
        corresponding labels or (ii) a vector of positions of the (then unique) 1-labels
        for each row of `scores`.

        """
        raise NotImplementedError()

    def _labels_as_matrix(self, scores, labels):
        """Reshapes `labels` into indexes if necessary.

        See `__call__`. This function converts case (ii) into case (i).
        """
        if labels.dim() == 2:
            return labels
        else:
            x = torch.zeros(
                scores.shape, device=self.config.get("job.device"), dtype=torch.float
            )
            x[range(len(scores)), labels] = 1.0
            return x

    def _labels_as_indexes(self, scores, labels):
        """Reshapes `labels` into matrix form if necessary and possible.

        See `__call__`. This function converts case (i) into case (ii). Throws an error
        if there is a row which does not have exactly one 1.

        """
        if labels.dim() == 1:
            return labels
        else:
            x = labels.nonzero()
            if not x[:, 0].equal(
                torch.arange(len(labels), device=self.config.get("job.device"))
            ):
                raise ValueError("exactly one 1 per row required")
            return x[:, 1]


class BCEWithLogitsKgeLoss(KgeLoss):
    def __init__(self, config, offset=0.0, bce_type=None, temperature=1.0, **kwargs):
        super().__init__(config)
        self._bce_type = bce_type
        if bce_type is None:
            reduction = "sum"
        elif bce_type is "mean":
            reduction = "none"
        elif bce_type is "self_adversarial":
            reduction = "none"
            self._temperature = temperature
        else:
            raise ValueError()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction=reduction, **kwargs)
        self._offset = offset

    def __call__(self, scores, labels, **kwargs):
        labels_as_matrix = self._labels_as_matrix(scores, labels)
        if self._offset != 0.0:
            scores = scores + self._offset
        losses = self._loss(scores.view(-1), labels_as_matrix.view(-1))
        if self._bce_type is None:
            return losses
        elif self._bce_type is "mean":
            labels = self._labels_as_indexes(scores, labels)
            losses = losses.view(scores.shape)
            losses_positives = losses[range(len(scores)), labels]
            losses_negatives = losses.sum(dim=1) - losses_positives

            return (
                losses_positives.sum() + losses_negatives.sum() / (scores.shape[1] - 1)
            ) / 2.0
        elif self._bce_type is "self_adversarial":
            labels = self._labels_as_indexes(scores, labels)
            negative_indexes = torch.nonzero(labels_as_matrix.view(-1) == 0.0)
            losses = losses.view(scores.shape)
            losses_positives = losses[range(len(scores)), labels]
            scores_negatives = (
                scores.detach()  # do not backprop adversarial weights
                .view(-1)[negative_indexes]
                .view((len(scores), scores.shape[1] - 1))
            )
            losses_negatives = losses.view(-1)[negative_indexes].view(
                (len(scores), scores.shape[1] - 1)
            )
            losses_negatives = (
                F.softmax(scores_negatives * self._temperature, dim=1)
                * losses_negatives
            ).sum(dim=1)

            return (losses_positives.sum() + losses_negatives.sum()) / 2.0
        else:
            raise NotImplementedError()


class KLDivWithSoftmaxKgeLoss(KgeLoss):
    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)
        self._celoss = torch.nn.CrossEntropyLoss(reduction=reduction, **kwargs)
        self._klloss = torch.nn.KLDivLoss(reduction=reduction, **kwargs)

    def __call__(self, scores, labels, **kwargs):
        if labels.dim() == 1:
            # Labels are indexes of positive classes, i.e., we are in a multiclass
            # setting. Then kl divergence can be computed more efficiently using
            # pytorch's CrossEntropyLoss. (since the entropy of the data distribution is
            # then 0 so kl divergence equals cross entropy)
            #
            # Gives same result as:
            #   labels = self._labels_as_matrix(scores, labels)
            # followed by using _klloss as below.
            return self._celoss(scores, labels)
        else:
            # label matrix; use KlDivLoss
            return self._klloss(
                F.log_softmax(scores, dim=1), F.normalize(labels.float(), p=1, dim=1)
            )


class SoftMarginKgeLoss(KgeLoss):
    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)
        self._loss = torch.nn.SoftMarginLoss(reduction=reduction, **kwargs)

    def __call__(self, scores, labels, **kwargs):
        labels = self._labels_as_matrix(scores, labels)
        labels = labels * 2 - 1  # expects 1 / -1 as label
        return self._loss(scores.view(-1), labels.view(-1))


class MarginRankingKgeLoss(KgeLoss):
    def __init__(self, config, margin, reduction="sum", **kwargs):
        super().__init__(config)
        self._device = config.get("job.device")
        self._train_type = config.get("train.type")
        self._loss = torch.nn.MarginRankingLoss(
            margin=margin, reduction=reduction, **kwargs
        )

    def __call__(self, scores, labels, **kwargs):
        # scores is (batch_size x num_negatives + 1)
        labels = self._labels_as_matrix(scores, labels)

        if "negative_sampling" in self._train_type:
            # Pair each 1 with the following zeros until next 1
            labels = labels.to(self._device).view(-1)
            pos_positives = labels.nonzero().view(-1)
            pos_negatives = (labels == 0).nonzero().view(-1)
            # repeat each positive score num_negatives times
            pos_positives = (
                pos_positives.view(-1, 1).repeat(1, kwargs["num_negatives"]).view(-1)
            )
            positives = scores.view(-1)[pos_positives].to(self._device).view(-1)
            negatives = scores.view(-1)[pos_negatives].to(self._device).view(-1)
            target = torch.ones(positives.size()).to(self._device)
            return self._loss(positives, negatives, target)

        elif self._train_type == "KvsAll":
            # TODO determine how to form pairs for margin ranking in KvsAll training
            # scores and labels are tensors of size (batch_size, num_entities)
            # Each row has 1s and 0s of a single sp or po tuple from training
            # How to combine them for pairs?
            # Each 1 with all 0s? Can memory handle this?
            raise NotImplementedError(
                "Margin ranking with KvsAll training not yet supported."
            )
        else:
            raise ValueError("train.type for margin ranking.")


class SEKgeLoss(KgeLoss):
    def __init__(self, config, reduction="sum", **kwargs):
        super().__init__(config)
        self._loss = torch.nn.MSELoss(reduction=reduction, **kwargs)

    def __call__(self, scores, labels, **kwargs):
        labels = self._labels_as_matrix(scores, labels)
        return self._loss(scores, labels)
