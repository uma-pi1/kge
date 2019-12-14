from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


class LookupEmbedder(KgeEmbedder):
    def __init__(
        self, config: Config, dataset: Dataset, configuration_key: str, vocab_size: int
    ):
        super().__init__(config, dataset, configuration_key)

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.normalize_with_grad = self.get_option("normalize.with_grad")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        # setup embedder
        self.embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse
        )

        # initialize weights
        init_ = self.get_option("initialize")
        try:
            init_args = self.get_option("initialize_args." + init_)
        except KeyError:
            init_args = self.get_option("initialize_args")

        # Automatically set arg a (lower bound) for uniform_ if not given
        if init_ == "uniform_" and "a" not in init_args:
            init_args["a"] = init_args["b"] * -1
            self.set_option("initialize_args.a", init_args["a"], log=True)

        self.initialize(self.embeddings.weight.data, init_, init_args)

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0:

            def normalize_embeddings(job):
                if self.normalize_with_grad:
                    self.embeddings.weight = torch.nn.functional.normalize(
                        self.embeddings.weight, p=self.normalize_p, dim=-1
                    )
                else:
                    with torch.no_grad():
                        self.embeddings.weight = torch.nn.Parameter(
                            torch.nn.functional.normalize(
                                self.embeddings.weight, p=self.normalize_p, dim=-1
                            )
                        )

            job.pre_batch_hooks.append(normalize_embeddings)

    def _embed(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def embed(self, indexes: Tensor) -> Tensor:
        return self._embed(self.embeddings(indexes.long()))

    def embed_all(self) -> Tensor:
        return self._embed(self.embeddings.weight)

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self.embeddings.weight
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    self.get_option("regularize_weight") / p * (parameters ** p).sum()
                ]
            else:
                # weighted Lp regularization
                unique_ids, counts = torch.unique(
                    kwargs["batch"]["triples"][:, kwargs["slot"]], return_counts=True
                )
                parameters = self.embeddings(unique_ids)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    self.get_option("regularize_weight")
                    / p
                    * (parameters ** p * counts.float().view(-1, 1)).sum()
                    # In contrast to unweighted Lp regulariztion, rescaling by number of
                    # triples is necessary here so that penalty term is correct in
                    # expectation
                    / len(kwargs["batch"]["triples"])
                ]
        else:  # unknown regularziation
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
