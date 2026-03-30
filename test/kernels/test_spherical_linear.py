"""Unit tests for the SphericalLinearKernel."""

import gpytorch
import pytest
import torch
from torch import Tensor

from src.kernels import rbf_poly_approx_factory, spherical_linear_factory


@pytest.fixture
def bounds() -> tuple[float, float]:
    """Fixture for input bounds."""
    return (0.0, 1.0)


@pytest.fixture
def xs(bounds: tuple[float, float]) -> Tensor:
    """Fixture for input tensor with shape [5, 3]."""
    torch.manual_seed(42)
    low, high = bounds
    res = torch.rand(5, 10) * (high - low) + low
    return res


class TestSphericalLinearKernel:
    """Test cases for the SphericalLinearKernel."""

    def test_equivalence_with_rbf_poly_approx_order_1(self, xs: Tensor, bounds: tuple[float, float]):
        """Test that SphericalLinearKernel produces the same output as RBFPolynomialApproximation with order=1."""
        n, d = xs.shape

        # Create both kernels with the same bounds and prior
        spherical_kernel = spherical_linear_factory(d=d, bounds=bounds, prior="dsp_unscaled")
        rbf_poly_kernel = rbf_poly_approx_factory(
            d=d, bounds=bounds, order=1, prior="dsp_unscaled", learn_coeffs=True, learn_offset=True
        )

        # Set the same hyperparameters for both kernels
        # Copy lengthscales from spherical to rbf_poly
        with torch.no_grad():
            rbf_poly_kernel.raw_lengthscale.copy_(spherical_kernel.raw_lengthscale)
            rbf_poly_kernel.raw_coeffs.copy_(spherical_kernel.raw_coeffs)
            rbf_poly_kernel.raw_offset.copy_(spherical_kernel.raw_glob_ls.item())

        # Compute kernel matrices
        K_spherical = spherical_kernel(xs, xs).to_dense()
        K_rbf_poly = rbf_poly_kernel(xs, xs).to_dense()

        # Check that they produce the same output
        torch.testing.assert_close(K_spherical, K_rbf_poly, rtol=1e-5, atol=1e-5)

    def test_inverse_stereographic_projection_produces_unit_norm(self, xs: Tensor, bounds: tuple[float, float]):
        """Test that inverse_stereographic_projection produces outputs with unit norm."""
        n, d = xs.shape

        # Create kernel
        kernel = spherical_linear_factory(d=d, bounds=bounds)

        # Apply inverse stereographic projection
        s = kernel.inverse_stereographic_projection(xs)

        # Check that all points have unit norm
        norms = torch.norm(s, dim=-1)
        torch.testing.assert_close(norms, torch.ones(n), rtol=1e-5, atol=1e-5)

    def test_stereographic_projection_inverse_recovers_input(self, xs: Tensor, bounds: tuple[float, float]):
        """Test that stereographic_projection followed by inverse_stereographic_projection recovers the input."""
        n, d = xs.shape

        # Create kernel
        kernel = spherical_linear_factory(d=d, bounds=bounds)

        # Apply inverse stereographic projection, then stereographic projection
        s = kernel.inverse_stereographic_projection(xs)
        xs_recovered = kernel.stereographic_projection(s)

        # Check that we recover the original input
        torch.testing.assert_close(xs_recovered, xs, rtol=1e-5, atol=1e-5)

    def test_prediction_strategy_consistency(self, bounds: tuple[float, float]):
        """Test that the custom LinearPredictionStrategy produces consistent posteriors."""
        # Set up test data
        torch.manual_seed(42)
        d = 10
        n_train = 20
        n_test = 5

        # Create train and test data within bounds
        low, high = bounds
        train_inputs = torch.rand(n_train, d) * (high - low) + low
        train_targets = torch.randn(n_train)
        test_inputs = torch.rand(n_test, d) * (high - low) + low

        # Create ExactGP with SphericalLinearKernel
        class SimpleGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = 0.001
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = spherical_linear_factory(d=d, bounds=bounds)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Test with lazily_evaluate_kernels = True
        with gpytorch.settings.lazily_evaluate_kernels(True):
            model_lazy = SimpleGP(train_inputs, train_targets)
            model_lazy.eval()

            with torch.no_grad():
                posterior_lazy = model_lazy.likelihood(model_lazy(test_inputs))
                mean_lazy = posterior_lazy.mean
                covar_lazy = posterior_lazy.covariance_matrix

        # Test with lazily_evaluate_kernels = False
        with gpytorch.settings.lazily_evaluate_kernels(False):
            model_eager = SimpleGP(train_inputs, train_targets)
            model_eager.eval()

            with torch.no_grad():
                posterior_eager = model_eager.likelihood(model_eager(test_inputs))
                mean_eager = posterior_eager.mean
                covar_eager = posterior_eager.covariance_matrix

        # Assert that posterior means and covariances match
        torch.testing.assert_close(
            mean_lazy,
            mean_eager,
            rtol=1e-2,
            atol=1e-3,
            msg="Posterior means should match between lazy and eager evaluation",
        )
        torch.testing.assert_close(
            covar_lazy,
            covar_eager,
            rtol=1e-2,
            atol=1e-3,
            msg="Posterior covariances should match between lazy and eager evaluation",
        )

    def test_raises_error_for_1d_input(self, bounds: tuple[float, float]):
        """Test that ard_num_dims=1 raises an error."""
        with pytest.raises(ValueError, match="ard_num_dims must be equal to the dimensionality"):
            spherical_linear_factory(d=1, bounds=bounds)

    def test_raises_error_for_missing_bounds_and_variances(self):
        """Test that missing both bounds and (input_variances, input_means) raises an error."""
        with pytest.raises(ValueError, match="Either \\(input_variances and input_means\\) or bounds must be provided"):
            spherical_linear_factory(d=3, bounds=None, input_variances=None, input_means=None)
