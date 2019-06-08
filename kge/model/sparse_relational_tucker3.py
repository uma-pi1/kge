from kge import Config, Dataset
from kge.model import RelationalTucker3

class SparseRelationalTucker3(RelationalTucker3):
    r"""Implementation of the ComplEx KGE model."""

    def post_score_loss_hook(self, epoch, epoch_step):
        return self.config.get(self.config.get("model") +
                               ".relation_embedder.l0_regularization") * \
        self._relation_embedder.penalty

    def post_update_trace_hook(self, trace_msg):
        trace_msg['density'] = self._relation_embedder.density
        return trace_msg

    def post_epoch_trace_hook(self, trace_msg):
        trace_msg['density'] = self._relation_embedder.density
        return trace_msg
