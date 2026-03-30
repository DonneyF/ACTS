"""Unit tests for the  RBF polynomial approximation kernel and associated functions."""

from typing import Literal

import gpytorch
import pytest
import torch
from torch import Tensor

from src.kernels._rbf_polynomial_approximation import RBFPolynomialApproximation, project_onto_unit_sphere


_ProjectionType = Literal["inverse_stereographic", "radial"]


@pytest.fixture
def max_sq_norm() -> Tensor:
    """Fixture for maximum squared norm."""
    return torch.tensor([[2.0]])


@pytest.fixture
def xs(max_sq_norm: torch.Tensor) -> Tensor:
    """Fixture for input tensor with shape [..., 5, 3]."""
    res = torch.randn(5, 3)
    max_norm = res.norm(dim=-1).max()
    res = res / max_norm * (0.95 * max_sq_norm.sqrt())
    return res


class TestProjectOntoSphere:
    """Test cases for the project_onto_unit_sphere function."""

    @pytest.mark.parametrize("projection", ["inverse_stereographic", "radial"])
    def test_shape_transformation(self, xs: Tensor, max_sq_norm: Tensor, projection: _ProjectionType):
        """Test that [..., 5, 3] input produces [..., 5, 4] output."""
        assert xs.shape == (5, 3)
        result = project_onto_unit_sphere(xs, max_sq_norm=max_sq_norm, projection=projection)
        assert result.shape == (5, 4), f"Expected shape (5, 4), got {result.shape}"

    @pytest.mark.parametrize("projection", ["inverse_stereographic", "radial"])
    def test_batch_shape_transformation(self, xs: Tensor, max_sq_norm: Tensor, projection: _ProjectionType):
        """Test that batched inputs work correctly."""
        x = xs.unsqueeze(0).repeat(2, 1, 1)  # Create batch of size 2
        result = project_onto_unit_sphere(x, max_sq_norm=max_sq_norm, projection=projection)
        assert result.shape == (2, 5, 4), f"Expected shape (2, 5, 4), got {result.shape}"

    @pytest.mark.parametrize("projection", ["inverse_stereographic", "radial"])
    def test_unit_norm_outputs(self, xs: Tensor, max_sq_norm: Tensor, projection: _ProjectionType):
        """Test that outputs have unit norm."""
        result = project_onto_unit_sphere(xs, max_sq_norm=max_sq_norm, projection=projection)

        # Check that all points lie on the sphere with radius sqrt(max_sq_norm)
        norms = torch.norm(result, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_radial_projection(self, xs: Tensor, max_sq_norm: Tensor):
        """Test specific properties of radial projection."""
        result = project_onto_unit_sphere(xs, max_sq_norm=max_sq_norm, projection="radial")
        # Check that the first 3 dimensions are only scaled
        torch.testing.assert_close(result[..., :3], xs / max_sq_norm.sqrt())

    def test_expected_sq_norm_inputs_inverse_stereographic(self, xs: Tensor, max_sq_norm: Tensor):
        """Test that inputs with norm = (max_sq_norm / 3).sqrt() are unchanged by the invbstereo projection."""
        # Create inputs with norm = (max_sq_norm / 3).sqrt()
        norm = (max_sq_norm / 3.0).sqrt()
        xs = xs / torch.norm(xs, dim=-1, keepdim=True) * norm  # Normalize to unit norm
        # Check that first dimensions are only scaled, and last dimension is zero
        result = project_onto_unit_sphere(xs, max_sq_norm=max_sq_norm, projection="inverse_stereographic")
        torch.testing.assert_close(result[..., :-1], xs / norm)
        torch.testing.assert_close(result[..., -1], torch.zeros(5))

    def test_invalid_projection_type(self, xs: Tensor, max_sq_norm: Tensor):
        """Test that invalid projection type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown projection type"):
            project_onto_unit_sphere(xs, max_sq_norm=max_sq_norm, projection="invalid")

    def test_kernel_properties_linear(self, xs: Tensor, max_sq_norm: Tensor):
        """Test kernel properties for linear approximation"""
        n, d = xs.shape
        bounds = (xs.min().item(), xs.max().item())
        kernel = RBFPolynomialApproximation(ard_num_dims=d, bounds=bounds, projection="inverse_stereographic", order=1)
        K = kernel(xs, xs).to_dense()
        torch.testing.assert_close(K.diagonal(), torch.ones(n))
        torch.testing.assert_close(K, K.T)
        torch.testing.assert_close(torch.linalg.eigvals(K).real >= 0.0, torch.ones(n, dtype=torch.bool))

    def test_kernel_properties_quadratic(self, xs: Tensor, max_sq_norm: Tensor):
        """Test kernel properties for quadratic approximation"""
        n, d = xs.shape
        bounds = (xs.min().item(), xs.max().item())
        kernel = RBFPolynomialApproximation(ard_num_dims=d, bounds=bounds, projection="inverse_stereographic", order=2)
        K = kernel(xs, xs).to_dense()
        torch.testing.assert_close(K.diagonal(), torch.ones(n))
        torch.testing.assert_close(K, K.T)
        torch.testing.assert_close(torch.linalg.eigvals(K).real >= 0.0, torch.ones(n, dtype=torch.bool))


class TestPredictionStrategyConsistency:
    """Test that custom prediction strategy doesn't affect posterior computation."""

    def test_posterior_consistency_with_prediction_strategy(self, with_outputscale: bool = False):
        """Test that posterior mean and covariances match between custom and default prediction strategies."""
        # Set up test data
        torch.manual_seed(42)
        d = 100
        n_train = 70
        n_test = 5

        # Create train and test data within bounds
        bounds = (-1.0, 1.0)
        train_inputs = torch.rand(n_train, d) * 2.0 - 1.0  # Scale to [-1, 1]
        train_targets = torch.randn(n_train)
        test_inputs = torch.rand(n_test, d) * 2.0 - 1.0  # Scale to [-1, 1]

        # Create ExactGP with constant mean and RBFPolynomialApproximation kernel (order=1)
        class SimpleGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = 0.001  # Set a fixed noise level
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = RBFPolynomialApproximation(ard_num_dims=d, bounds=bounds, order=1)
                if with_outputscale:
                    self.covar_module = gpytorch.kernels.ScaleKernel(self.covar_module)
                    self.covar_module.outputscale = 2.0  # Set a fixed outputscale

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # Test with lazily_evaluate_kernels = True (should use custom prediction strategy)
        with gpytorch.settings.lazily_evaluate_kernels(True):
            model_lazy = SimpleGP(train_inputs, train_targets)
            model_lazy.eval()

            with torch.no_grad():
                posterior_lazy = model_lazy.likelihood(model_lazy(test_inputs))
                mean_lazy = posterior_lazy.mean
                covar_lazy = posterior_lazy.covariance_matrix

        # Test with lazily_evaluate_kernels = False (should use default prediction strategy)
        with gpytorch.settings.lazily_evaluate_kernels(False):
            model_eager = SimpleGP(train_inputs, train_targets)
            model_eager.eval()

            with torch.no_grad():
                posterior_eager = model_eager.likelihood(model_eager(test_inputs))
                mean_eager = posterior_eager.mean
                covar_eager = posterior_eager.covariance_matrix

        # Assert that posterior means and covariances match
        torch.testing.assert_close(
            mean_lazy, mean_eager, msg="Posterior means should match between lazy and eager evaluation"
        )
        torch.testing.assert_close(
            covar_lazy, covar_eager, msg="Posterior covariances should match between lazy and eager evaluation"
        )

    def test_posterior_consistency_with_prediction_strategy_with_outputscale(self):
        """Test that posterior mean and covariances match between custom and default prediction strategies with
        outputscale."""
        self.test_posterior_consistency_with_prediction_strategy(with_outputscale=True)
