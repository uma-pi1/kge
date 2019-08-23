import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel


class FreexScorer(RelationalScorer):
    """Mixing matrix with ComplEx pattern of entries, but entries free.

    Initial sandbox model for experimentation with sparsity patterns.
    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        o_emb1, o_emb2 = (t.contiguous() for t in o_emb.chunk(2, dim=1))
        s_all = torch.cat((s_emb, s_emb), dim=1)
        o_all = torch.cat((o_emb, o_emb2, o_emb1), dim=1)

        if combine == "spo":
            out = (s_all * o_all * p_emb).sum(dim=1)
        elif combine == "sp*":
            out = (s_all * p_emb).mm(o_all.transpose(0, 1))
        elif combine == "*po":
            out = (p_emb * o_all).mm(s_all.transpose(0, 1))
        else:
            raise ValueError('cannot handle combine="{}".format(combine)')

        return out.view(n, -1)


class Freex(KgeModel):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # set relation embedder dimensionality
        if self.get_option("relation_embedder.dim") < 0:
            self.config.set(
                self.configuration_key + ".relation_embedder.dim",
                2 * self.get_option("entity_embedder.dim"),
                log=True,
            )

        # auto initialize such that scores have unit variance
        if self.get_option("entity_embedder.initialize") == "auto_initialization" and \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            # TODO these calculations may not be correct anymore (they are for ComplEx)
            #
            # Var[score] = 4*(dim/2)*var_e^2*var_r, where var_e/var_r are the variances
            # of the entries in the embeddings and (dim/2) is the embedding size in
            # complex space
            #
            # Thus we set var_e=var_r=(1.0/(2.0*dim))^(1/3)
            dim = self.get_option("entity_embedder.dim")
            std = math.pow(1.0 / 2.0 / dim, 1.0 / 6.0)

            config.set(
                self.configuration_key + ".entity_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".entity_embedder.initialize_args",
                {"mean": 0.0, "std": std},
                log=True,
            )
            config.set(
                self.configuration_key + ".relation_embedder.initialize",
                "normal_",
                log=True,
            )
            config.set(
                self.configuration_key + ".relation_embedder.initialize_args",
                {"mean": 0.0, "std": std},
                log=True,
            )
        elif self.get_option("entity_embedder.initialize") == "auto_initialization" or \
                self.get_option("relation_embedder.initialize") == "auto_initialization":
            raise ValueError("Both entity and relation embedders must be set to auto_initialization "
                             "in order to use it.")

        super().__init__(
            config,
            dataset,
            FreexScorer(config, dataset),
            configuration_key=configuration_key,
        )
