import torch


class KgeLoss:
    """ Wraps torch loss functions """

    @staticmethod
    def create(config):
        """ Factory method for loss creation """
        if config.get("train.loss") == "ce":
            return torch.nn.CrossEntropyLoss(reduction="mean")
        elif config.get("train.loss") == "bce":
            return torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=None)
        elif config.get("train.loss") == "kl":
            return torch.nn.KLDivLoss(reduction="mean")
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.loss")

    # mxn score matrix, mxn label matrix
    # if labels is none, assume first column positive, rest negative
    # TODO make it callable
    def loss(self, scores, labels=None):
        raise NotImplementedError()


class BceKgeLoss(KgeLoss):
    def __init__(self):
        super().__init(self)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=None)

    def loss(self, scores, labels=None):
        if labels is None:
            # TODO construct label matrix
            raise NotImplementedError("TODO")
        else:
            self.loss(scores.view(-1), labels.view(-1))
