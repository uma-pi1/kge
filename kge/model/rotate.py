import torch
import math
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


# TODO sp_ and _po scoring with RotatE leads to *large* intermediate results. It's
# unclear whether this can be fixed. Expect out-of-memory errors when using RotatE with
# 1vsAll or KvsAll training. To do validation/evaluation, you may want to set
# eval.chunk_size.
class RotatEScorer(RelationalScorer):
    r"""Implementation of the RotatE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # determine real and imaginary part
        s_emb_re, s_emb_im = torch.chunk(s_emb, 2, dim=1)
        o_emb_re, o_emb_im = torch.chunk(o_emb, 2, dim=1)

        # convert from radians to points on complex unix ball
        p_emb_re, p_emb_im = torch.cos(p_emb), torch.sin(p_emb)

        if combine == "spo":
            # compute the difference vector (s*p-t)
            sp_emb_re, sp_emb_im = hadamard_complex(
                s_emb_re, s_emb_im, p_emb_re, p_emb_im
            )
            diff_re, diff_im = diff_complex(sp_emb_re, sp_emb_im, o_emb_re, o_emb_im)

            # compute the absolute values for each (complex) element of the difference
            # vector
            diff_abs = abs_complex(diff_re, diff_im)

            # now take the norm of the absolute values of the difference vector
            out = -norm_nonnegative(diff_abs, dim=1, p=self._norm)
        elif combine == "sp_":
            # as above, but pair each sp-pair with each object
            sp_emb_re, sp_emb_im = hadamard_complex(
                s_emb_re, s_emb_im, p_emb_re, p_emb_im
            )  # sp x dim
            diff_re, diff_im = pairwise_diff_complex(
                sp_emb_re, sp_emb_im, o_emb_re, o_emb_im
            )  # sp x o x dim
            diff_abs = abs_complex(diff_re, diff_im)  # sp x o x dim
            out = -norm_nonnegative(diff_abs, dim=2, p=self._norm)
        elif combine == "_po":
            # compute the complex conjugate (cc) of the relation vector and perform
            # inverse rotation on tail. This uses || s*p - o || = || s - cc(p)*o || for
            # a rotation p.
            p_emb_im = -p_emb_im
            po_emb_re, po_emb_im = hadamard_complex(
                p_emb_re, p_emb_im, o_emb_re, o_emb_im
            )  # po x dim
            diff_re, diff_im = pairwise_diff_complex(
                po_emb_re, po_emb_im, s_emb_re, s_emb_im
            )  # po x s x dim
            diff_abs = abs_complex(diff_re, diff_im)  # po x s x dim
            out = -norm_nonnegative(diff_abs, dim=2, p=self._norm)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class RotatE(KgeModel):
    r"""Implementation of the RotatE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        self._init_configuration(config, configuration_key)
        if self.get_option("entity_embedder.dim") % 2 != 0:
            raise ValueError(
                "RotatE requires embeddings of even dimensionality"
                " (got {})".format(self.get_option("entity_embedder.dim"))
            )
        if self.get_option("relation_embedder.dim") < 0:
            self.set_option(
                "relation_embedder.dim",
                self.get_option("entity_embedder.dim") // 2,
                log=True,
            )
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RotatEScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self._normalize_phases = self.get_option("normalize_phases")

    @torch.no_grad()
    def normalize_phases(self):
        out = self.get_p_embedder()._embeddings.weight.data

        # normalize phases so that they lie in [-pi,pi]
        # TODO this is a hack that assumes that we use a lookup embedder

        # first shift phases by pi
        out = out + math.pi

        # compute the modulo (result then in [0,2*pi))
        out = torch.remainder(out, 2.0 * math.pi)

        # shift back
        out = out - math.pi

        # write back the updated embeddings
        self.get_p_embedder()._embeddings.weight.data[:] = out[:]

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)

        if self._normalize_phases and isinstance(job, TrainingJob):
            from kge.model import LookupEmbedder

            if not isinstance(self.get_p_embedder(), LookupEmbedder):
                raise ValueError(
                    "RotatE currently supports normalize_phases=True "
                    "only when a lookup embedder is used for relations; "
                    "current relation embedder is "
                    f"{self.get_option('relation_embedder.type')} "
                    "however"
                )

            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self.normalize_phases())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self.normalize_phases())


@torch.jit.script
def pairwise_sum(X, Y):
    """Compute pairwise sum of rows of X and Y.

    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) + Y


@torch.jit.script
def pairwise_diff(X, Y):
    """Compute pairwise difference of rows of X and Y.

    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) - Y


@torch.jit.script
def pairwise_hadamard(X, Y):
    """Compute pairwise Hadamard product of rows of X and Y.

    Returns tensor of shape len(X) x len(Y) x dim."""
    return X.unsqueeze(1) * Y


@torch.jit.script
def hadamard_complex(x_re, x_im, y_re, y_im):
    "Hadamard product for complex vectors"
    result_re = x_re * y_re - x_im * y_im
    result_im = x_re * y_im + x_im * y_re
    return result_re, result_im


@torch.jit.script
def pairwise_hadamard_complex(x_re, x_im, y_re, y_im):
    "Pairwise Hadamard product for complex vectors"
    result_re = pairwise_hadamard(x_re, y_re) - pairwise_hadamard(x_im, y_im)
    result_im = pairwise_hadamard(x_re, y_im) + pairwise_hadamard(x_im, y_re)
    return result_re, result_im


@torch.jit.script
def diff_complex(x_re, x_im, y_re, y_im):
    "Difference of complex vectors"
    return x_re - y_re, x_im - y_im


@torch.jit.script
def pairwise_diff_complex(x_re, x_im, y_re, y_im):
    "Pairwise difference of complex vectors"
    return pairwise_diff(x_re, y_re), pairwise_diff(x_im, y_im)


@torch.jit.script
def abs_complex(x_re, x_im):
    "Compute magnitude of given complex numbers"
    x_re_im = torch.stack((x_re, x_im), dim=0)  # dim0: real, imaginary
    return torch.norm(x_re_im, dim=0)  # sqrt(real^2+imaginary^2)


@torch.jit.script
def norm_nonnegative(x, dim: int, p: float):
    "Computes lp-norm along dim assuming that all inputs are non-negative."
    if p == 1.0:
        # speed up things for this common case. We known that the inputs are
        # non-negative here.
        return torch.sum(x, dim=dim)
    else:
        return torch.norm(x, dim=dim, p=p)
