"""Unit tests for the MaximizerDistribution."""

import math

import gpytorch
import torch
from botorch.models import SingleTaskGP

from src.acquisitions._maximizer_distribution import SphericalLinearMaximizerDistribution, sphere_log_prob
from src.kernels import spherical_linear_factory


class TestMaximizerDistribution:
    """Test cases for the MaximizerDistribution."""

    def test_sphere_log_prob(self):
        """Test that _sphere_log_prob integrates to approximately 1 over the unit sphere in 2D."""

        mu = torch.tensor([0.5, -0.2])
        chol_prec = torch.tensor([[1.5, 0.0], [0.2, 0.3]])

        # Create polar grid over unit sphere in 2D
        # In 2D, sphere is a circle parameterized by angle theta
        n_theta = 10
        theta = torch.linspace(0, 2 * math.pi, n_theta + 1)[:-1]

        # Points on unit circle
        ss = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (n_theta, 2)

        # Evaluate log_prob on all points
        log_prob_ss = sphere_log_prob(ss, mu, chol_prec)

        # Other approximate integral
        max_stdv = 10.0
        n_alpha = 100
        alphas = torch.linspace(0, max_stdv, n_alpha)
        ws = alphas[..., None, None] * ss
        log_prob_ws = torch.distributions.MultivariateNormal(
            loc=mu,
            precision_matrix=(chol_prec @ chol_prec.mT),
        ).log_prob(ws)
        log_prob_ss_approx = torch.logsumexp(log_prob_ws, dim=-2) + math.log(max_stdv / n_alpha)

        # Check that integral is close to 1
        torch.testing.assert_close(log_prob_ss, log_prob_ss_approx, rtol=5e-2, atol=5e-2)

    def test_beta_dist(self):
        """Test that _beta_dist gives correct posterior moments"""
        n = 2
        m = 20
        d = 3
        train_X = torch.randn(n, d)
        train_Y = torch.randn(n, 1)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=spherical_linear_factory(d=d),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
        model.eval()
        maximizer_distribution = SphericalLinearMaximizerDistribution(spherical_linear_gp=model)
        beta_mean, beta_chol_prec = maximizer_distribution._beta_dist

        # Using beta_dist, let's predict posterior moments on new test points
        test_X = torch.randn(m, d)
        test_S = model.covar_module.inverse_stereographic_projection(test_X)
        terms = model.covar_module.coeffs
        term0_sqrt = terms[0].sqrt()
        term1_sqrt = terms[1].sqrt()
        test_S_plus_const = torch.cat([test_S * term1_sqrt, term0_sqrt.expand_as(test_S[..., :1])], dim=-1)
        test_mean = (test_S_plus_const @ beta_mean).squeeze(-1)
        test_root_covar = torch.linalg.solve_triangular(beta_chol_prec, test_S_plus_const.mT, upper=False).mT
        test_covar = test_root_covar @ test_root_covar.mT

        # Compare to actual moments from the model
        test_dist = model(test_X)
        torch.testing.assert_close(test_mean, test_dist.mean)
        torch.testing.assert_close(test_covar, test_dist.covariance_matrix)

    def test_beta_sample(self):
        """Test that _beta_sample gives correct posterior moments via sampling"""
        torch.manual_seed(42)
        n = 2
        m = 20
        d = 3
        num_samples = 10000

        train_X = torch.randn(n, d)
        train_Y = torch.randn(n, 1)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=spherical_linear_factory(d=d),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
        model.mean_module.constant.data.fill_(0.0)
        model.eval()
        maximizer_distribution = SphericalLinearMaximizerDistribution(spherical_linear_gp=model)
        beta_samples = maximizer_distribution._beta_sample(torch.Size([num_samples]))

        # Using beta_dist, let's predict posterior moments on new test points
        test_X = torch.randn(m, d)
        test_S = model.covar_module.inverse_stereographic_projection(test_X)
        terms = model.covar_module.coeffs
        term0_sqrt = terms[0].sqrt()
        term1_sqrt = terms[1].sqrt()
        test_S_plus_const = torch.cat([test_S * term1_sqrt, term0_sqrt.expand_as(test_S[..., :1])], dim=-1)
        fn_samples = (test_S_plus_const @ beta_samples) + model.mean_module.constant

        # Get sample moments
        fn_mean_sample = fn_samples.mean(dim=-3)
        fn_covar_sample = ((fn_samples - fn_mean_sample) @ (fn_samples - fn_mean_sample).mT).mean(dim=-3)

        # Get true moments
        fn_dist = model(test_X)
        torch.testing.assert_close(fn_mean_sample, fn_dist.mean.unsqueeze(-1), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(fn_covar_sample, fn_dist.covariance_matrix, rtol=3e-2, atol=3e-2)

    def test_inverse_stereographic_projection_jacobian(self):
        """Test that _inverse_stereographic_projection_jacobian matches autograd computation."""
        torch.manual_seed(42)

        d = 5
        n = 2
        batch_size = 3

        # Create training data and model
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.randn(n, 1, dtype=train_X.dtype)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=spherical_linear_factory(d=d),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
        model.to(dtype=train_X.dtype)
        model.eval()

        maximizer_distribution = SphericalLinearMaximizerDistribution(spherical_linear_gp=model)

        # Create test points on the sphere (unit norm vectors)
        test_s = torch.randn(batch_size, d + 1, dtype=torch.double)
        test_s = test_s / test_s.norm(dim=-1, keepdim=True)

        # Compute Jacobian using the implementation
        jac_impl = maximizer_distribution._inverse_stereographic_projection_jacobian(test_s)

        # Compute Jacobian using autograd
        # We need to compute d(inverse_stereographic_projection)/dx
        # where x is computed from s via stereographic_projection
        test_x = maximizer_distribution._spherical_linear_kernel.stereographic_projection(test_s)
        test_x.requires_grad_(True)

        def inverse_stereo_fn(x):
            return maximizer_distribution._spherical_linear_kernel.inverse_stereographic_projection(x)

        jac_autograd = torch.autograd.functional.jacobian(inverse_stereo_fn, test_x)
        # jac_autograd is (batch_size, d+1, batch_size, d)
        # Extract the diagonal batch elements to get (batch_size, d+1, d)
        jac_autograd = torch.stack([jac_autograd[i, :, i, :] for i in range(batch_size)])

        # Compare
        torch.testing.assert_close(jac_impl, jac_autograd, rtol=1e-6, atol=1e-6)

    def test_rsample_concentrates_on_optimum(self):
        """Test that samples from the maximizer distribution concentrate near the optimal point."""
        # Lock random seed
        torch.manual_seed(3)

        d = 3
        n = 2
        n_samples = 10000

        # Create d+1 training points
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.randn(n, 1, dtype=train_X.dtype)

        # Create model with very little observation noise
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=spherical_linear_factory(d=d),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
        model.likelihood.noise = 1e-3
        model.to(dtype=train_X.dtype)
        model.eval()

        with torch.no_grad():
            maximizer_distribution = SphericalLinearMaximizerDistribution(spherical_linear_gp=model)
            maximizer_samples = maximizer_distribution.rsample(sample_shape=torch.Size([n_samples]))  # (n_samples, d)
            avg_maximizer = maximizer_samples.mean(dim=-2)

        with torch.no_grad():
            num_grid_points = 1000
            grid_points = torch.randn(num_grid_points, d, dtype=train_X.dtype)
            fn_samples = model.posterior(grid_points).rsample(sample_shape=torch.Size([n_samples])).squeeze(-1)
            # (n_samples, num_grid_points)
            approx_maximizer_samples = grid_points[fn_samples.argmax(dim=-1)]
            approx_avg_maximizer = approx_maximizer_samples.mean(dim=-2)

        # The sample mean should be very close to the optimal point
        torch.testing.assert_close(avg_maximizer, approx_avg_maximizer, rtol=1e-1, atol=1e-1)

    def test_log_prob(self):
        """Test that log_prob is higher at the empirical mode than at other points."""
        torch.manual_seed(42)

        d = 3
        n = 2
        n_samples = 10000

        # Create training data and model
        train_X = torch.rand(n, d, dtype=torch.double)
        train_Y = torch.randn(n, 1, dtype=train_X.dtype)
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            mean_module=gpytorch.means.ConstantMean(),
            covar_module=spherical_linear_factory(d=d),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
        model.to(dtype=train_X.dtype)
        model.eval()

        maximizer_distribution = SphericalLinearMaximizerDistribution(spherical_linear_gp=model)

        # Draw samples and compute empirical mode (mean of samples)
        with torch.no_grad():
            maximizer_samples = maximizer_distribution.rsample(sample_shape=torch.Size([n_samples]))
            empirical_mode = maximizer_samples.mean(dim=0)

        # Compute log_prob at empirical mode
        log_prob_at_mode = maximizer_distribution.log_prob(empirical_mode)

        # Generate random test points in the input space
        n_test_points = 20
        test_points = torch.randn(n_test_points, d, dtype=torch.double)

        # Compute log_prob at test points
        log_prob_at_test_points = maximizer_distribution.log_prob(test_points)

        # Assert that log_prob at empirical mode is higher than at all test points
        assert (log_prob_at_mode > log_prob_at_test_points).all()
