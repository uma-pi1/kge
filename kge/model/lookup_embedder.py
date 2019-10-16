import torch.nn
import torch.nn.functional

from kge.model import KgeEmbedder
from kge.util.misc import round_to_points


class LookupEmbedder(KgeEmbedder):
    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key)

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.normalize_with_grad = self.get_option("normalize.with_grad")
        self.regularize = self.check_option("regularize", ["", "l1", "l2", "l3"])
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

        # Automatically set arg b for uniform_ if not given
        # TODO can we avoid the hacky if "uniform_"?
        if init_ == "uniform_" and "b" not in init_args:
            init_args["b"] = init_args["a"] * -1
            self.config.set(
                configuration_key + ".initialize_args.b", init_args["b"], log=True
            )

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

    def prepare_job(self, job, **kwargs):
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

    def _embed(self, embeddings):
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def embed(self, indexes):
        return self._embed(self.embeddings(indexes.long()))

    def embed_all(self):
        return self._embed(self.embeddings.weight)

    def penalty(self, **kwargs):
        # TODO factor out to a utility method
        if self.regularize == "" or self.get_option("regularize_args.weight") == 0.0:
            return super().penalty(**kwargs)
        elif self.regularize == "l1":
            if self.get_option("regularize_args.weighted"):
                result = super().penalty(**kwargs)
                if "batch" in kwargs and "triples" in kwargs["batch"]:
                    unique_ids, counts = torch.unique(
                        kwargs["batch"]["triples"][:, kwargs["slot"]],
                        return_counts=True
                    )
                    parameters = self.embeddings(unique_ids)
                    result += [
                        self.get_option("regularize_args.weight")
                        * (torch.abs(parameters)*counts.float().view(-1, 1)).sum()
                        / parameters.size(0)
                    ]
                return result
            else:
                return super().penalty(**kwargs) + [
                    self.get_option("regularize_args.weight")
                    * self.embeddings.weight.norm(p=1)
                ]
        elif self.regularize == "l2":
            if self.get_option("regularize_args.weighted"):
                result = super().penalty(**kwargs)
                if "batch" in kwargs and "triples" in kwargs["batch"]:
                    unique_ids, counts = torch.unique(
                        kwargs["batch"]["triples"][:, kwargs["slot"]],
                        return_counts=True
                    )
                    parameters = self.embeddings(unique_ids)
                    result += [
                        self.get_option("regularize_args.weight")
                        * (parameters**2*counts.float().view(-1, 1)).sum()
                        / parameters.size(0)
                    ]
                return result
            else:
                return super().penalty(**kwargs) + [
                    self.get_option("regularize_args.weight")
                    * self.embeddings.weight.norm(p=2) ** 2
                ]
        elif self.regularize == "l3":
            if self.get_option("regularize_args.weighted"):
                result = super().penalty(**kwargs)
                if "batch" in kwargs and "triples" in kwargs["batch"]:
                    unique_ids, counts = torch.unique(
                        kwargs["batch"]["triples"][:, kwargs["slot"]],
                        return_counts=True
                    )
                    parameters = self.embeddings(unique_ids)
                    result += [
                        self.get_option("regularize_args.weight")
                        * (torch.abs(parameters)**3*counts.float().view(-1, 1)).sum()
                        / parameters.size(0)
                    ]
                return result
            else:
                # As in CP-N3 paper, Eq. (4): Timothée Lacroix, Nicolas Usunier, Guillaume
                # Obozinski. Canonical Tensor Decomposition for Knowledge Base Completion.
                # ICML 2018. https://arxiv.org/abs/1806.07297
                return super().penalty(**kwargs) + [
                    self.get_option("regularize_args.weight")
                    * self.embeddings.weight.norm(p=3) ** 3
                ]
        else:
            raise ValueError("unknown penalty")
