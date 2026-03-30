"""
Thompson sampling acquisition functions
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal, Optional

import botorch
import gpytorch.settings as gpts
import torch
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling as PathwiseTS
from botorch.generation.utils import _flip_sub_unique
from jaxtyping import Float
from torch import Tensor
from torch.quasirandom import SobolEngine

from src.acquisitions.base import Acquisition
from src.models.jacobian import JacobianRBFGP


def get_sampler_ctx(sampler: str = "cholesky", no_grad=True) -> ExitStack:
    """
    Constructs a context with appropriate GPyTorch settings for selected samplers

    Args:
        sampler: name of sampling scheme. One of "cholesky", "ciq", "lanczos", "rff" (default: cholesky)
        no_grad: Whether to additionally include torch.no_grad context (default: True)

    Returns:
        An exit stack containing the sampler contexts
    """
    es = ExitStack()

    if sampler == "cholesky":
        es.enter_context(gpts.max_cholesky_size(float("inf")))
    elif sampler == "ciq":
        es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
        es.enter_context(gpts.max_cholesky_size(0))
        es.enter_context(gpts.ciq_samples(True))
        es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
        es.enter_context(gpts.num_contour_quadrature(15))
    elif sampler == "lanczos":
        es.enter_context(gpts.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True))
        es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
        es.enter_context(gpts.max_cholesky_size(0))
        es.enter_context(gpts.ciq_samples(False))
    elif sampler == "rff":
        # Can only be used when the GP kernel is also RFF
        es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    if no_grad:
        es.enter_context(torch.no_grad())

    return es


def generate_candidate_points(
    X: Float[Tensor, "n d"],
    Y: Float[Tensor, "n 1"],
    bounds: Float[Tensor, "2 d"],
    n_cand: int,
    candidate_policy: Literal["sobol", "uniform", "raasp"],
    seed: int,
) -> Float[Tensor, "n_cand d"]:
    """
    Generates a set of candidate points within bounds

    Args:
        X: Domain data
        Y: Evaluated points
        bounds: Ambient space to generate candidates from
        n_cand: number of candidate points
        candidate_policy: One of 'sobol', 'uniform', 'raasp'
        seed: Random seed

    Returns:
        A num_candidates x d array of candidate points
    """
    dim = X.shape[-1]

    if candidate_policy == "raasp":
        x_center = X[Y.argmax(), :].clone()
        # Default behavior used by TurBO. Perturb near the incumbent within the trust region.
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        pert = sobol.draw(n_cand).to(X)
        pert = bounds[0] + (bounds[1] - bounds[0]) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_cand, dim).to(X) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_cand, dim).clone()
        X_cand[mask] = pert[mask]
    elif candidate_policy == "sobol":
        sobol = SobolEngine(dim, scramble=True, seed=seed)
        pert = sobol.draw(n_cand).to(dtype=torch.float64, device=X.device)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * pert
    elif candidate_policy == "uniform":
        X_cand = torch.rand(n_cand, dim).to(X)
        X_cand = bounds[0] + (bounds[1] - bounds[0]) * X_cand
    else:
        raise NotImplementedError(f"Unknown candidate generation policy {candidate_policy}")

    return X_cand


class MaxPosteriorSampling(torch.nn.Module):
    """
    A modification of the botorch MaxPosteriorSampling class to further collect the posterior samples
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        replacement: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.replacement = replacement

    def forward(self, X, num_samples: int = 1, observation_noise: bool = False):
        """Samples from the posterior maximizer distribution."""
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X, samples, num_samples: int = 1):
        """Maximizes samples from the posterior distribution."""
        obj = samples.squeeze(-1)
        if self.replacement:
            idcs = torch.argmax(obj, dim=-1)
        else:
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )

        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)

        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        X_samp = torch.gather(Xe, -2, idcs)
        acq_score = torch.gather(obj, dim=-1, index=idcs[:, :1])

        return X_samp, acq_score


@dataclass
class CandidateThompsonSampling(Acquisition):
    """
    Simple Thompson sampling through the use of candidate points
    """

    candidate_policy: Literal["sobol", "uniform", "raasp"] = "raasp"
    replacement: bool = False  # When q>1, whether TS are obtained with replacement of the candidate points

    def argmax(  # noqa: D102
        self,
        model,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
        q: int,
        bounds: Float[Tensor, "2 d"],
        options: Optional[dict] = None,
        **kwargs,
    ) -> Float[Tensor, "q d"]:
        n_cand = options["n_cand"]
        sampler = options["ts_sampler"]

        X_cand = generate_candidate_points(
            X=X,
            Y=Y,
            candidate_policy=self.candidate_policy,
            n_cand=n_cand,
            seed=options.get("seed", 0) if options else 0,
            bounds=bounds,
        )

        with get_sampler_ctx(sampler):
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=self.replacement)
            X_next, values = thompson_sampling(X_cand, num_samples=q)

        return X_next


class PathwiseThompsonSampling(Acquisition):
    """
    Pathwise Thompson Sampling
    """

    max_batch_size: int = None

    def argmax(  # noqa: D102
        self,
        model,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
        q: int,
        bounds: Float[Tensor, "2 d"],
        options: Optional[dict] = None,
        **kwargs,
    ) -> Float[Tensor, "q d"]:
        # For q > 1, try our best to optimize all q samples simultaneously, backing down slowly if we run out of VRAM.
        if self.max_batch_size is None:
            self.max_batch_size = q

        while True:
            try:
                X_next = []
                # values = []

                n = q // self.max_batch_size
                remainder = q % self.max_batch_size
                batches = [self.max_batch_size] * n
                if remainder > 0:
                    batches.append(remainder)

                for q_ in batches:
                    TS = PathwiseTS(model)
                    X_i, values_i = botorch.optim.optimize_acqf(TS, bounds=bounds, q=q_, **options.get("optimize_acqf"))
                    X_next.append(X_i)
                    # values.append(values_i)

                X_next = torch.vstack(X_next).to(X)
                # values = torch.vstack(values).to(X)

                return X_next
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if self.max_batch_size == 1:
                    raise e

                self.max_batch_size -= 1

@dataclass
class AdaptiveCandidateThompsonSampling(Acquisition):

    perturb: Literal['standard', 'L1', 'L2', 'L3', 'TopK', 'softmax'] = 'L2'
    tmvn: bool = False

    def argmax(  # noqa: D102
        self,
        model,
        X: Float[Tensor, "n d"],
        Y: Float[Tensor, "n 1"],
        q: int,
        bounds: Float[Tensor, "2 d"],
        options: Optional[dict] = None,
        **kwargs,
    ) -> Float[Tensor, "q d"]:
        n_cand = options["n_cand"]
        sampler = options["ts_sampler"]

        if not isinstance(model, JacobianRBFGP):
            raise NotImplementedError(f'Cannot use TSGradientRAASP on {type(model)}')

        dim = X.shape[-1]
        if bounds is None:
            bounds = torch.tensor([[0.0, 1.0]] * dim).to(X).T

        with get_sampler_ctx(sampler):
            x0 = X[Y.argmax(), :].clone()
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_cand).to(X)

            # Gradient sample
            gradfx0_samples = model.sample_gradfx0(tmvn=self.tmvn, num_samples=q) # M D 1

            # Build the batch sequentially
            X_nexts = []
            for gradfx0 in gradfx0_samples:
                gradfx0 = gradfx0 # 1 by D
                # Find the ACTS search space, intersect with trust region bounds.
                tr_ub = torch.where(gradfx0 >= 0, 1, x0)
                tr_lb = torch.where(gradfx0 < 0, 0, x0)

                tr_ub = torch.minimum(tr_ub, bounds[1])
                tr_lb = torch.maximum(tr_lb, bounds[0])


                # Move Sobol points into this space
                X_sobol_acts = tr_lb + (tr_ub - tr_lb) * pert

                # Create a perturbation mask
                match self.perturb:
                    case 'standard':
                        prob_perturb = min(20.0 / dim, 1.0)
                    case 'L1':
                        prob_perturb = torch.minimum(20.0 * gradfx0.abs() / gradfx0.norm(p=1), torch.tensor([1.]).to(X))
                    case 'L2':
                        prob_perturb = torch.minimum(20.0 * gradfx0.square() / gradfx0.norm(p=2).square(), torch.tensor([1.]).to(X))
                    case 'L3':
                        prob_perturb = torch.minimum(20.0 * gradfx0.abs().pow(3) / gradfx0.norm(p=3).pow(3), torch.tensor([1.]).to(X))
                    case 'TopK':
                        prob_perturb = torch.zeros_like(gradfx0).put_(gradfx0.abs().topk(20).indices, torch.ones(20).to(X))
                    case 'softmax':
                        prob_perturb = 20 * torch.softmax(gradfx0.abs(), dim=1)
                    case 'all':
                        prob_perturb = torch.minimum(dim * gradfx0.square() / gradfx0.norm(p=2).square(), torch.tensor([1.]).to(X))
                    case 'tenth':
                        prob_perturb = torch.minimum(dim / 10 * gradfx0.square() / gradfx0.norm(p=2).square(), torch.tensor([1.]).to(X))
                    case _:
                        raise NotImplementedError(f'{self.perturb} not implemented.')

                mask = torch.rand(n_cand, dim, dtype=torch.float64, device=X.device) <= prob_perturb
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=X.device)] = 1

                # Create candidate points from the perturbations and the mask
                X_cand = x0.expand(n_cand, dim).clone()
                X_cand[mask] = X_sobol_acts[mask]

                # Sample on posterior, conditioned by gradient
                fX_cand = model.fx_conditioned_on_gradfx0(x0, gradfx0, X_cand)

                ts = MaxPosteriorSampling(model=model, replacement=False)  # We just instantiate to call maximize_samples

                samples = fX_cand.rsample(sample_shape=torch.Size([1]))
                X_next, values = ts.maximize_samples(X_cand, samples, num_samples=1)

                X_nexts.append(X_next)

                del fX_cand

            X_next = torch.cat(X_nexts, dim=0)

            return X_next