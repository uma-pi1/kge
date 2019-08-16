import torch.nn
import torch.nn.functional

from kge.model import KgeEmbedder


class LookupEmbedder(KgeEmbedder):
    def __init__(self, config, dataset, configuration_key, vocab_size):
        super().__init__(config, dataset, configuration_key)

        # read config
        # self.dropout = torch.nn.Dropout(self.get_option("dropout"))
        self.normalize_p = self.get_option("normalize.p")
        self.normalize_with_grad = self.get_option("normalize.with_grad")
        self.regularize = self.check_option("regularize", ["", "l1", "l2", "l3"])
        self.regularize_args = self.get_option("regularize_args")
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        # setup embedder
        self.embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
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
            self.config.set(configuration_key + ".initialize_args.b", init_args["b"], log=True)

        self.initialize(
            self.embeddings.weight.data,
            init_,
            init_args
        )

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

        self.penalized_params_cache = None

    def prepare_job(self, job, **kwargs):
        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0:
            def normalize_embeddings(job):
                if self.normalize_with_grad:
                    self.embeddings.weight = torch.nn.functional.\
                        normalize(self.embeddings.weight, p=self.normalize_p, dim=-1)
                else:
                    with torch.no_grad():
                        self.embeddings.weight = torch.nn.Parameter(torch.nn.functional.
                        normalize(self.embeddings.weight, p=self.normalize_p, dim=-1))
            job.pre_batch_hooks.append(normalize_embeddings)

    def _embed(self, embeddings):
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def embed(self, indexes):
        self.penalized_params_cache = self._embed(self.embeddings(indexes.long()))
        return self.penalized_params_cache

    def embed_all(self):
        return self._embed(self.embeddings.weight)

    def penalty(self, **kwargs):
        # TODO factor out to a utility method
        if self.regularize == "" or self.regularize_args['weight'] == 0.0:
            return super().penalty(**kwargs)
        elif self.regularize == "l1":
            if not self.regularize_args['sparse']:
                return super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.embeddings.weight.norm(p=1)
                ]
            else:
                result = super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.penalized_params_cache.norm(p=1)
                ]
                self.penalized_params_cache = None
                return result
        elif self.regularize == "l2":
            if not self.regularize_args['sparse']:
                return super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.embeddings.weight.norm(p=2) ** 2
                ]
            else:
                result = super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.penalized_params_cache.norm(p=2) ** 2
                ]
                self.penalized_params_cache = None
                return result
        elif self.regularize == "l3":
            # As in CP-N3 paper, Eq. (4): Timothée Lacroix, Nicolas Usunier, Guillaume
            # Obozinski. Canonical Tensor Decomposition for Knowledge Base Completion.
            # ICML 2018. https://arxiv.org/abs/1806.07297
            if not self.regularize_args['sparse']:
                return super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.embeddings.weight.norm(p=3) ** 3
                ]
            else:
                result = super().penalty(**kwargs) + [
                    self.regularize_args['weight'] * self.penalized_params_cache.norm(p=3) ** 3
                ]
                self.penalized_params_cache = None
                return result
        else:
            raise ValueError("unknown penalty")
