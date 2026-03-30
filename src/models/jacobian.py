
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.probability import LinearEllipticalSliceSampler
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import Mean

from linear_operator import LinearOperator
import gpytorch
import linear_operator
import torch
from botorch.models import SingleTaskGP
from gpytorch.distributions import MultivariateNormal as MVN
from jaxtyping import Float
from torch import Tensor
from gpytorch.constraints import GreaterThan
from functools import partial

from src.priors import LogNormalPrior

class SingleTaskGradGP(SingleTaskGP):

    def gradfx0(self: gpytorch.models.GP, x0: Float[Tensor, "... D"]) -> Float[MVN, "... D"]:
        """
        Compute the (posterior) distribution of :math:`∇f(x_0)`,

        :param f: The GP model of :math:`f`.
        :param x0: The point :math:`x_0` at which to compute the gradient.
        :return: Distribution of :math:`∇f(x_0) | y`, where :math:`y` represents the training responses.
        """

        def _cross_covar_func(x1: Float[Tensor, "... D"], x2: Float[Tensor, "... D"]) -> Float[Tensor, ""]:
            """
            Given two inputs `x1` and `x2`, compute the cross covariance `Cov[f(x1); f(x2) | y]` (summed across batch dims).
            We will use this function to get the gradient of the GP at a given point by passing in `x1=x0` and `x2=x0`.
            Autograd will treat these two instances as different inputs, so we can compute the
            cross-derivatives of the covariance rather than the Hessian of the variance.
            """
            return self(torch.stack([x1, x2], dim=-2)).covariance_matrix[..., 0, 1].sum()

        def gradfx0_fx0_cross_covar_fn(x0: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
            """
            Given an input `x0`, compute the cross covariance `Cov[f(x0); ∇f(x_0) | y]`.
            """
            x0_clone = x0.clone().detach().requires_grad_(True)
            return torch.autograd.functional.vjp(
                partial(_cross_covar_func, x2=x0), inputs=x0_clone, create_graph=True
            )[1]

        # Compute the mean of ∇f(x_0)
        x0_ = x0.detach().clone().requires_grad_(True)
        fx0_mean = self(x0_.unsqueeze(-2)).mean.sum()
        gradfx0_mean = torch.autograd.grad(fx0_mean, x0_)[0] if fx0_mean.requires_grad else torch.zeros_like(x0)

        # Compute the covariance of ∇f(x_0)
        x0_ = x0.detach().clone().requires_grad_(True)
        gradfx0_covar = torch.autograd.functional.jacobian(
            gradfx0_fx0_cross_covar_fn, inputs=x0_, vectorize=False
        )

        return MVN(gradfx0_mean, gradfx0_covar)

    def sample_gradfx0(self, x0: Float[Tensor, "... D"]) -> Float[Tensor, "... 1 D"]:
        gradfx0 = self.gradfx0(x0)
        gradfx0_post_mean = gradfx0.mean
        gradfx0_post_chol_covar = gradfx0.lazy_covariance_matrix.cholesky(upper=False)

        gradfx0_sample: Float[Tensor, "... D 1"] = (
                gradfx0_post_mean + gradfx0_post_chol_covar @ torch.randn_like(gradfx0_post_mean)
        )

        return gradfx0_sample.unsqueeze(0)


    def fx_conditioned_on_gradfx0(
            self: gpytorch.models.GP,
            x0: Float[Tensor, "... D"],
            grad_fx0: Float[Tensor, "... D"],
            x: Float[Tensor, "... M D"]
    ) -> Float[MVN, "... M"]:
        """
        Compute the (posterior) distribution of :math:`f(x)`
        *conditioned* on :math:`∇f(x_0)` for some `x_0`

        :param f: The GP model of :math:`f`.
        :param x0: The point :math:`x_0` at which we have gradient information to condition on
        :param grad_fx0: The gradient :math:`∇f(x_0)` to condition on.
        :param x: Inputs to compute GP on.
        :return: Distribution of :math:`f(x) | ∇f(x_0), y`, where :math:`y` represents the training responses.
        """

        def _cross_covar_fn(x0: Float[Tensor, "... D"], x: Float[Tensor, "... M D"]) -> Float[Tensor, "... M"]:
            """ Compute the cross covariance `Cov[f(x0); f(x) | y] """
            return self(torch.cat([x0.unsqueeze(-2), x])).covariance_matrix[..., 0, 1:]

        # Helper terms
        _fx: Float[MVN, "... M"] = self(x)
        _gradfx0: Float[MVN, "... D"] = self.gradfx0(x0)

        # Mean terms needed for conditoning
        fx_mean: Float[Tensor, "... M 1"] = _fx.mean.unsqueeze(-1)
        gradfx0_mean: Float[Tensor, "... D 1"] = _gradfx0.mean.unsqueeze(-1)

        # Covariance terms needed for conditioning
        fx_covar: Float[Tensor, "... M M"] = _fx.covariance_matrix
        gradfx0_fx_cross_covar: Float[Tensor, "... D M"] = torch.autograd.functional.jacobian(
            partial(_cross_covar_fn, x=x), inputs=x0
        )
        gradfx0_covar: Float[LinearOperator, "... D D"] = _gradfx0.lazy_covariance_matrix

        # compute `_interp_term = gradfx0_covar^{-1} gradfx0_fx_cross_covar`
        # which will be useful for the conditioned mean and variance
        _interp_term: Float[Tensor, "... D M"] = gradfx0_covar.solve(gradfx0_fx_cross_covar.mT)

        # Return distribution on f(x) conditioned on gradient
        return MVN(
            (fx_mean + _interp_term.mT @ (grad_fx0.unsqueeze(-1) - gradfx0_mean)).squeeze(-1),
            linear_operator.to_linear_operator(fx_covar - gradfx0_fx_cross_covar @ _interp_term)
        )

class JacobianRBFGP(SingleTaskGP):
    """
    RBF GP with line searching to obtain Thompson samples along a sampled gradient direction.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        dsp: bool = True, # Whether to use Dimensional-Scaled Prior or not
        **kwargs
    ):
        # Hvarfner defaults
        d = train_X.shape[-1]
        if dsp:
            batch_shape = torch.Size()
            noise_prior = LogNormalPrior(loc=-4.0, scale=1.0, device=train_X.device)
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
            )
            base_kernel = get_covar_module_with_dim_scaled_prior(use_rbf_kernel=True, ard_num_dims=d)
            covar_module = ScaleKernel(base_kernel)
        else:
            likelihood = GaussianLikelihood()
            base_kernel = RBFKernel(ard_num_dims=d)
            covar_module = ScaleKernel(base_kernel)

        # Fix the outputscale to 1
        covar_module.outputscale = torch.tensor(1.0, device=train_X.device, dtype=torch.float64)
        covar_module.raw_outputscale.requires_grad_(False)

        super().__init__(
            train_X, train_Y, likelihood=likelihood, covar_module=covar_module, mean_module=gpytorch.means.ZeroMean()#, outcome_transform=None
        )

        # Set the taylor approximation point, the incumbent as the maximizer of the posterior mean
        # Y.argmax() is equivalent in the noiseless case.
        self.register_buffer("x0", train_X[..., train_Y.argmax(), :].unsqueeze(-2))

    def posterior_gradfx0_fx0(self) -> Float[MVN, "... D+1"]:
        r"""
        Return the posterior distribution over :math:`\nabla f(x_0)` and :math:`f(x_0),
        where :math:`x_0` is the taylor approximation point (i.e.:attr:`self.x0`).

        :return: The joint posterior distribution over :math:`\nabla f(x_0)` and :math:`f(x_0)`,
        stored as a D+1-dimensional multivariate Normal distribution.
        """
        x: Float[Tensor, "... N D"] = self.train_inputs[0]  # Train inputs
        x0: Float[Tensor, "... 1 D"] = self.x0.unsqueeze(0)  # Incumbant input
        y: Float[Tensor, "... N 1"] = self.train_targets.unsqueeze(-1)  # Train targets
        d: int = x.size(-1)
        sq_lengthscale: Float[Tensor, "... 1 D"] = self.covar_module.base_kernel.lengthscale.square()
        outputscale: Float[Tensor, "... 1 D"] = self.covar_module.outputscale

        # Compute ktilde_x_x (k_xx with observational noise)
        mu_x = self.mean_module(self.train_inputs[0])
        k_x_x: Float[Tensor, "... N N"] = self.covar_module(x, x)
        p_fx: Float[MVN, "... N"] = MVN(mu_x, linear_operator.to_linear_operator(k_x_x))
        ktilde_x_x: Float[Tensor, "... N N"] = self.likelihood(p_fx, self.train_inputs).lazy_covariance_matrix

        # Compute cross term: [k_x_x0; k_x_gradx0]
        k_x_x0: Float[Tensor, "... N 1"] = self.covar_module(x, x0).to_dense()
        k_x_gradx0: Float[Tensor, "... N D"] = (x - x0) / sq_lengthscale * k_x_x0
        cross: Float[Tensor, "... N D+1"] = torch.cat([k_x_gradx0, k_x_x0], dim=-1)

        # Compute k_joint (joint prior covariance of f(x0) and ∇f(x0))
        #  = outputscale * [1; 0] @ [0; I / lengthscale^2]
        eye = torch.eye(d + 1, device=x0.device, dtype=x0.dtype)  # Shape [D, D]
        k_joint = eye * outputscale
        k_joint[..., :d, :d] = k_joint[..., :d, :d] / sq_lengthscale

        # Compute interp term: ktilde_x_x^{-1} @ cross
        # which will be used in the posterior mean and covariance calculations
        with linear_operator.settings.max_cholesky_size(1e4):
            interp_term: Float[Tensor, "... N D+1"] = ktilde_x_x.solve(cross)

        # Posterior mean
        # First term is joint posterior mean of [∇f(x0); f(x0)]
        post_mean: Float[Tensor, "... D+1 1"] = interp_term.mT @ y

        # Posterior covariance
        # First term is joint posterior covariance Cholesky factor of [∇f(x0); f(x0)]
        post_chol_covar: Float[Tensor, "... D+1 D+1"] = linear_operator.utils.cholesky.psd_safe_cholesky(k_joint - interp_term.mT @ cross)
        post_covar: Float[LinearOperator, "... D+1 D+1"] = linear_operator.operators.RootLinearOperator(post_chol_covar)

        # Done!
        return MVN(post_mean.squeeze(-1), post_covar)

    def sample_gradfx0(self, tmvn=False, num_samples: int = 1) -> Float[Tensor, "... M D"]:
        x: Float[Tensor, "... N D"] = self.train_inputs[0]  # Train inputs
        x0: Float[Tensor, "... 1 D"] = self.x0.unsqueeze(0)  # Incumbant input

        # Constants
        n, d = x.shape[-2:]

        # Get components of posterior distribution on ∇f(x0)
        gradfx0_fx0_post: Float[MVN, "... D+1"] = self.posterior_gradfx0_fx0()
        gradfx0_post_mean: Float[Tensor, "... D 1"] = gradfx0_fx0_post.mean[..., :d].unsqueeze(-1)
        gradfx0_post_chol_ccovar: Float[Tensor, "... D D"] = gradfx0_fx0_post.lazy_covariance_matrix.cholesky(
            upper=False
        ).to_dense()[..., :d, :d]

        if tmvn:
            # Draw a posterior sample of ∇f(x0) from a TMVN with rectangular boundaries
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            on_lower_boundary = x0.squeeze(-2) <= 1e-5
            on_upper_boundary = x0.squeeze(-2) >= 1 - 1e-5
            A = torch.cat([-eye[on_lower_boundary], eye[on_upper_boundary]], dim=0)
            b = torch.zeros(A.size(-2), 1, device=x.device, dtype=x.dtype)
            sampler = LinearEllipticalSliceSampler(
                inequality_constraints=(A, b),
                mean=gradfx0_post_mean.squeeze(-1),
                covariance_root=gradfx0_post_chol_ccovar,
                burnin=500,
                thinning=10,
                num_chains=1,
            )
            gradfx0_sample: Float[Tensor, "... M D 1"] = sampler.draw(n=num_samples)
        else:
            gradfx0_sample: Float[Tensor, "... D 1"] = (
                    gradfx0_post_mean + gradfx0_post_chol_ccovar @ torch.randn(d, num_samples).to(x)
            ).mT

        return gradfx0_sample

    def fx_conditioned_on_gradfx0(
            self: gpytorch.models.GP,
            x0: Float[Tensor, "... D"],
            grad_fx0: Float[Tensor, "... 1 D"],
            Xtest: Float[Tensor, "... M D"]
    ) -> Float[MVN, "... M"]:
        r"""
        :return: The posterior distribution of $f(x_{test_1}), \ldots, f(x_{test_M})$
        """
        x: Float[Tensor, "... N D"] = self.train_inputs[0]  # Train inputs
        y: Float[Tensor, "... N 1"] = self.train_targets.unsqueeze(-1)  # Train targets
        sq_lengthscale: Float[Tensor, "... 1 D"] = self.covar_module.base_kernel.lengthscale.square()
        sigma_sq: Float[Tensor, "... 1 1"] = self.likelihood.noise_covar.noise.unsqueeze(-1)
        outputscale: Float[Tensor, "... 1"] = self.covar_module.outputscale

        # Constants
        n, d = x.shape[-2:]
        m = Xtest.size(0)

        ## STEP 1: compute a posterior sample of ∇f(x0)
        gradfx0_sample = grad_fx0.T

        # Done by refactoring X test to be passed in here
        x0 = x0.unsqueeze(0)

        ## STEP 3: compute the joint posterior distribution of [f(x0), f(xtest), ∇f(x0)] | y
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        x_joint: Float[Tensor, "... N+1+M D"] = torch.cat([x, x0, Xtest], dim=-2)
        k_joint_joint: Float[Tensor, "... N+1+M N+1+M"] = self.covar_module(x_joint, x_joint).to_dense()
        k_joint_x0: Float[Tensor, "... N+1+M 1"] = k_joint_joint[..., :, n:n + 1]
        k_joint_gradx0: Float[Tensor, "... N+1+M D"] = (x_joint - x0) / sq_lengthscale * k_joint_x0
        k_gradx0_gradx0 = eye * outputscale / sq_lengthscale

        del k_joint_x0, x_joint

        # Complete covariance matrix between [f(X), f(x0), f(xtest), ∇f(x0)]
        # k_full: Float[Tensor, "... N+1+M+D  N+1+M+D"] = torch.cat(
        #     [
        #         torch.cat([k_joint_joint, k_joint_gradx0], dim=-1),
        #         torch.cat([k_joint_gradx0.mT, k_gradx0_gradx0], dim=-1),
        #     ],
        #     dim=-2,
        # )
        # Instead of torch.cat, preallocate then fill.
        n_total = k_joint_joint.size(-1)
        total_rows = n_total + d  # same for columns
        k_full = torch.empty(*k_joint_joint.shape[:-2], total_rows, total_rows, device=k_joint_joint.device, dtype=k_joint_joint.dtype)
        k_full[..., :n_total, :n_total] = k_joint_joint
        k_full[..., n_total:, :n_total] = k_joint_gradx0.mT
        del k_joint_joint
        k_full[..., :n_total, n_total:] = k_joint_gradx0
        del k_joint_gradx0
        k_full[..., n_total:, n_total:] = k_gradx0_gradx0
        del k_gradx0_gradx0

        # Now compute [f(x0), f(xtest), ∇f(x0)] | y
        with linear_operator.settings.max_cholesky_size(1e6):
            cov_y_others: Float[Tensor, "... N 1+M+D"] = k_full[..., :n, n:]  # Cov(y; [f(x0), f(xtest), ∇f(x0)])
            eye: Float[Tensor, "N N"] = torch.eye(n, device=x.device, dtype=x.dtype)
            ktilde_x_x: Float[LinearOperator, "... N N"] = linear_operator.to_linear_operator(
                k_full[..., :n, :n] + sigma_sq * eye
            )  # (k(X, X) + σ^2 I)
            # Interp term: (k(X, X) + σ^2 I)^{-1} Cov(y; [f(x0), f(xtest), ∇f(x0)])
            interp_term: Float[Tensor, "... N 1+M+D"] = ktilde_x_x.solve(cov_y_others)

        # Posterior terms of [f(x0), f(xtest), ∇f(x0)] | y
        fxtest_fx0_gradfx0_post_mean: Float[Tensor, "... 1+M+D 1"] = interp_term.mT @ y
        # fxtest_fx0_gradfx0_post_covar: Float[Tensor, "... 1+M+D 1+M+D"] = (
        #         k_full[..., n:, n:] - interp_term.mT @ cov_y_others
        # )

        # More memory efficient to perform calculation in-place
        fxtest_fx0_gradfx0_post_covar = k_full[..., n:, n:].clone()
        fxtest_fx0_gradfx0_post_covar.addmm_(interp_term.mT, cov_y_others, beta=1.0, alpha=-1.0)

        del k_full, interp_term, cov_y_others, ktilde_x_x

        ## STEP 4: using the joint distribution of [f(x0), f(xtest), ∇f(x0)] | y,
        # further condition f(xtest) on the sample of ∇f(x0)
        # to get f(xtest) | y, ∇f(x0)

        # Condition this term on a sample of grad_fx0
        with linear_operator.settings.max_cholesky_size(1e6):
            # Compute the posterior mean and covariance cholesky factor of [f(xtest_1), \ldots, f(xtest_M)]
            cov_fxtest_fxtest: Float[Tensor, "... M M"] = fxtest_fx0_gradfx0_post_covar[..., 1:-d, 1:-d]
            cov_gradfx0_fxtest: Float[Tensor, "... D M"] = fxtest_fx0_gradfx0_post_covar[..., -d:, 1:-d]
            cov_gradfx0_gradfx0: Float[LinearOperator, "... D D"] = linear_operator.to_linear_operator(
                fxtest_fx0_gradfx0_post_covar[..., -d:, -d:]
            )
            interp_term: Float[Tensor, "... D M"] = cov_gradfx0_gradfx0.solve(cov_gradfx0_fxtest)
        del fxtest_fx0_gradfx0_post_covar
        # Posterior of f(xtest) | y, ∇f(x0)
        fxtest_post_mean: Float[Tensor, "... M 1"] = fxtest_fx0_gradfx0_post_mean[..., 1:-d, :] + (
                interp_term.mT @ (gradfx0_sample - fxtest_fx0_gradfx0_post_mean[..., -d:, :])
        )
        del fxtest_fx0_gradfx0_post_mean
        # fxtest_post_covar: Float[Tensor, "... M M"] = linear_operator.to_linear_operator(
        #     cov_fxtest_fxtest - interp_term.mT @ cov_gradfx0_fxtest
        # )
        # More memory efficient to perform calculation in-place
        prod_linop = linear_operator.operators.MatmulLinearOperator(interp_term.mT, cov_gradfx0_fxtest)
        fxtest_post_covar = cov_fxtest_fxtest - prod_linop
        del cov_fxtest_fxtest, interp_term, cov_gradfx0_fxtest

        fxtest: Float[MVN, "... M"] = MVN(fxtest_post_mean.squeeze(-1), fxtest_post_covar)

        return fxtest

    def line_search(
            self,
            gradfx0: Float[Tensor, "... 1 D"],
            Xtest: Float[Tensor, "... M D"],
    ) -> tuple[Float[MVN, "... M"], Float[Tensor, "... M"]]:
        r"""
        Performs a line search of $f(x)$ along through Thompson samples of xtest$.

        If normalize is True, the thompson samples are $f(x_0 + \alpha_i ∇f(x_0) / || ∇f(x_0) ||)$.

        :return: A tuple containing:
            (1) the posterior distribution of $f(x_{test_1}), \ldots, f(x_{test_M})$
            (2) the expected square error, $E[(f(xtest) - f(x0) - (x - x0)^T @ ∇f(x0))^2]$
        """
        x: Float[Tensor, "... N D"] = self.train_inputs[0]  # Train inputs
        x0: Float[Tensor, "... 1 D"] = self.x0.unsqueeze(0)  # Incumbant input
        y: Float[Tensor, "... N 1"] = self.train_targets.unsqueeze(-1)  # Train targets
        sq_lengthscale: Float[Tensor, "... 1 D"] = self.covar_module.base_kernel.lengthscale.square()
        sigma_sq: Float[Tensor, "... 1 1"] = self.likelihood.noise_covar.noise.unsqueeze(-1)
        outputscale: Float[Tensor, "... 1"] = self.covar_module.outputscale

        # Constants
        n, d = x.shape[-2:]
        m = Xtest.size(0)

        ## STEP 1: compute a posterior sample of ∇f(x0)
        gradfx0_sample = gradfx0.T

        ## STEP 2: using the posterior sample of ∇f(x0),
        # choose a set of candidate points xtest that are conical combinations of x0 and ∇f(x0)
        # clamped to the boundaries of the trust region

        # Done by refactoring X test to be passed in here

        ## STEP 3: compute the joint posterior distribution of [f(x0), f(xtest), ∇f(x0)] | y
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        x_joint: Float[Tensor, "... N+1+M D"] = torch.cat([x, x0, Xtest], dim=-2)
        k_joint_joint: Float[Tensor, "... N+1+M N+1+M"] = self.covar_module(x_joint, x_joint).to_dense()
        k_joint_x0: Float[Tensor, "... N+1+M 1"] = k_joint_joint[..., :, n:n + 1]
        k_joint_gradx0: Float[Tensor, "... N+1+M D"] = (x_joint - x0) / sq_lengthscale * k_joint_x0
        k_gradx0_gradx0 = eye * outputscale / sq_lengthscale

        # Complete covariance matrix between [f(X), f(x0), f(xtest), ∇f(x0)]
        k_full: Float[Tensor, "... N+1+M+D  N+1+M+D"] = torch.cat(
            [
                torch.cat([k_joint_joint, k_joint_gradx0], dim=-1),
                torch.cat([k_joint_gradx0.mT, k_gradx0_gradx0], dim=-1),
            ],
            dim=-2,
        )

        # Now compute [f(x0), f(xtest), ∇f(x0)] | y
        with linear_operator.settings.max_cholesky_size(1e6):
            cov_y_others: Float[Tensor, "... N 1+M+D"] = k_full[..., :n, n:]  # Cov(y; [f(x0), f(xtest), ∇f(x0)])
            eye: Float[Tensor, "N N"] = torch.eye(n, device=x.device, dtype=x.dtype)
            ktilde_x_x: Float[LinearOperator, "... N N"] = linear_operator.to_linear_operator(
                k_joint_joint[..., :n, :n] + sigma_sq * eye
            )  # (k(X, X) + σ^2 I)
            # Interp term: (k(X, X) + σ^2 I)^{-1} Cov(y; [f(x0), f(xtest), ∇f(x0)])
            interp_term: Float[Tensor, "... N 1+M+D"] = ktilde_x_x.solve(cov_y_others)

        # Posterior terms of [f(x0), f(xtest), ∇f(x0)] | y
        fxtest_fx0_gradfx0_post_mean: Float[Tensor, "... 1+M+D 1"] = interp_term.mT @ y
        fxtest_fx0_gradfx0_post_covar: Float[Tensor, "... 1+M+D 1+M+D"] = (
                k_full[..., n:, n:] - interp_term.mT @ cov_y_others
        )

        ## STEP 4: using the joint distribution of [f(x0), f(xtest), ∇f(x0)] | y,
        # further condition f(xtest) on the sample of ∇f(x0)
        # to get f(xtest) | y, ∇f(x0)

        # Condition this term on a sample of grad_fx0
        with linear_operator.settings.max_cholesky_size(1e6):
            # Compute the posterior mean and covariance cholesky factor of [f(xtest_1), \ldots, f(xtest_M)]
            cov_fxtest_fxtest: Float[Tensor, "... M M"] = fxtest_fx0_gradfx0_post_covar[..., 1:-d, 1:-d]
            cov_gradfx0_fxtest: Float[Tensor, "... D M"] = fxtest_fx0_gradfx0_post_covar[..., -d:, 1:-d]
            cov_gradfx0_gradfx0: Float[LinearOperator, "... D D"] = linear_operator.to_linear_operator(
                fxtest_fx0_gradfx0_post_covar[..., -d:, -d:]
            )
            interp_term: Float[Tensor, "... D M"] = cov_gradfx0_gradfx0.solve(cov_gradfx0_fxtest)

        # Posterior of f(xtest) | y, ∇f(x0)
        fxtest_post_mean: Float[Tensor, "... M 1"] = fxtest_fx0_gradfx0_post_mean[..., 1:-d, :] + (
                interp_term.mT @ (gradfx0_sample - fxtest_fx0_gradfx0_post_mean[..., -d:, :])
        )
        fxtest_post_covar: Float[Tensor, "... M M"] = linear_operator.to_linear_operator(
            cov_fxtest_fxtest - interp_term.mT @ cov_gradfx0_fxtest
        )
        fxtest: Float[MVN, "... M"] = MVN(fxtest_post_mean.squeeze(-1), fxtest_post_covar)

        ## STEP 5: again using the joint distribution of [f(x0), f(xtest), ∇f(x0)] | y,
        # compute the distribution of Δ | y
        # where Δ = f(xtest) - f(x0) - (x - x0)^T @ ∇f(x0)

        # We will multiply the [f(x0), f(xtest), ∇f(x0)] | y moments by
        # lin_combo_weights = [-1, 1, -(x - x0)] to get the distribution of Δ | y
        lin_combo_weights: Float[Tensor, "... M 1+M+D"] = torch.cat([
            torch.full_like(fxtest_post_mean, fill_value=-1.0),  # -1 weight for f(x0)
            torch.eye(m, device=x.device, dtype=x.dtype),  # 1 weight for f(xtest)
            (x0 - Xtest)  # -(x - x0) weight for ∇f(x0)
        ], dim=-1)
        # Multiply these lin_combo_weights by the moments of [f(x0), f(xtest), ∇f(x0)] | y
        delta_mean: Float[Tensor, "... M 1"] = lin_combo_weights @ fxtest_fx0_gradfx0_post_mean
        # delta_covar: Float[Tensor, "... M M"] = lin_combo_weights @ fxtest_fx0_gradfx0_post_covar @ lin_combo_weights.mT
        # # We now have the delta distribution
        # delta: Float[MVN, "... M"] = MVN(delta_mean.squeeze(-1), linear_operator.to_linear_operator(delta_covar))

        # Fast computation of variance by exploiting matrix structure
        B = fxtest_fx0_gradfx0_post_covar

        B11: Float[Tensor, "... 1 1"] = B[0, 0]
        B12: Float[Tensor, "... 1 M"] = B[0, 1:1+m]
        B13: Float[Tensor, "... 1 D"] = B[0, 1+m:1+m+d]
        B22: Float[Tensor, "... M M"] = B[1:1+m, 1:1+m]
        B23: Float[Tensor, "... M D"] = B[1:1+m, 1+m:1+m+d]
        B33: Float[Tensor, "... D D"] = B[1+m: 1+m+d, 1+m:1+m+d]
        v = x0 - Xtest

        delta_variance = B11 - 2 * B12 + torch.diagonal(B22) + (v @ B33 * v).sum(dim=-1) -2 * (B13 @ v.mT) + 2 * (B23 * v).sum(dim=-1)

        # Check equality
        # assert torch.isclose(delta_variance, torch.diagonal(delta_covar)).all().item()

        expected_square_error = delta_mean.squeeze(-1).square() + delta_variance

        # Done!
        return fxtest, expected_square_error
