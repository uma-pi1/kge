import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, RelationalScorer
from kge.job import Job
from torch.nn import functional as F


def score_kl(mean, ent_cov, rel_cov):
    return -(
        torch.sum((1 / rel_cov) * ent_cov, dim=-1)
        + torch.sum(mean * mean * (1 / rel_cov), dim=-1)
        - torch.sum(torch.log(ent_cov), dim=-1)
        + torch.sum(torch.log(rel_cov), dim=-1)
    )


def score_el(mean, ent_cov, rel_cov):
    return -(
        torch.sum(mean * mean * (1 / (ent_cov + rel_cov)), dim=-1)
        + torch.sum(torch.log(ent_cov + rel_cov), dim=-1)
    )


class KG2EScorer(RelationalScorer):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        score_metric = self.get_option('score_function')
        if score_metric.upper() == 'KL':
            self.score_func = score_kl
        elif score_metric.upper() == 'EL':
            self.score_func = score_el
        else:
            raise ValueError('No score function implemented for metric {}'.format(score_metric))

    def _score(self, s_emb, p_emb, o_emb):
        s_mean, s_cov = torch.chunk(s_emb, 2, dim=1)
        p_mean, p_cov = torch.chunk(p_emb, 2, dim=1)
        o_mean, o_cov = torch.chunk(o_emb, 2, dim=1)
        # compute the overall mean and entity covariance matrix
        mean = s_mean - p_mean - o_mean
        ent_cov = s_cov + o_cov

        return self.score_func(mean, ent_cov, p_cov)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):

        n = p_emb.size(0)
        if combine == "spo":
            # n = n_s = n_p = n_o
            out = self._score(s_emb, p_emb, o_emb)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class KG2E(KgeModel):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        self.dim = self.get_option("entity_embedder.dim")
        self.set_option("entity_embedder.dim", self.dim * 2, log=True)
        if self.get_option("relation_embedder.dim") < 0:
            self.set_option("relation_embedder.dim", self.dim * 2, log=True)
        self.c_min = self.get_option('c_min')
        self.c_max = self.get_option('c_max')
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=KG2EScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self._apply_constraints()

    @torch.no_grad()
    def _apply_constraints(self):
        # bound entity mean vectors to 1
        ent_mean = self._entity_embedder._embeddings.weight.data[:, :self.dim]
        ent_mean = ent_mean.where(torch.norm(ent_mean, dim=1, keepdim=True) < 1, F.normalize(ent_mean, dim=1))
        self._entity_embedder._embeddings.weight.data[:, :self.dim] = ent_mean

        # bound relation mean vectors to 1
        rel_mean = self._relation_embedder._embeddings.weight.data[:, :self.dim]
        rel_mean = rel_mean.where(torch.norm(rel_mean, dim=1, keepdim=True) < 1, F.normalize(rel_mean, dim=1))
        self._relation_embedder._embeddings.weight.data[:, :self.dim] = rel_mean

        # clamp entity cov between c_min and c_max
        self._entity_embedder._embeddings.weight.data[:, self.dim:].clamp_(self.c_min, self.c_max)

        # clamp relation cov between c_min and c_max
        self._relation_embedder._embeddings.weight.data[:, self.dim:].clamp_(self.c_min, self.c_max)

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob
        super().prepare_job(job, **kwargs)

        if isinstance(job, TrainingJob):
            job.post_batch_hooks.append(lambda job: self._apply_constraints())
