"""Test cases for the fit_linear_mll function."""

import logging
import unittest

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.fit.fit_linear_mll import fit_linear_mll

# Adjust imports based on your project structure
from src.kernels.linear import LinearKernel


logger = logging.getLogger(__name__)


class TestFitLinearMLL(unittest.TestCase):
    """Test cases for the fit_linear_mll function."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Ensure reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def test_compare_fit_linear_mll_vs_botorch_n_greater_than_d(self):
        """Compare fit_linear_mll vs botorch.fit_gpytorch_mll for n > d."""
        # Generate synthetic data: y = w^T x + noise
        self.n = 50
        self.d = 4
        self.X = torch.rand(self.n, self.d)
        self.weights = torch.randn(self.d, 1)
        self.Y = self.X @ self.weights + 0.05 * torch.randn(self.n, 1)
        self.bounds = (0.0, 1.0)

        # 1. Setup Model for fit_linear_mll
        # We use LinearKernel directly as covar_module. SingleTaskGP will use it as is.
        # We initialize with a specific lengthscale prior/constraint via the class defaults,
        # but we'll ensure they start at the same point.
        likelihood1 = GaussianLikelihood()
        covar1 = LinearKernel(ard_num_dims=None, bounds=self.bounds)
        model1 = SingleTaskGP(self.X, self.Y, likelihood=likelihood1, covar_module=covar1)

        # 2. Setup Model for botorch fit (Baseline)
        likelihood2 = GaussianLikelihood()
        covar2 = LinearKernel(ard_num_dims=None, bounds=self.bounds)
        model2 = SingleTaskGP(self.X, self.Y, likelihood=likelihood2, covar_module=covar2)

        # Ensure both models start with identical parameters
        model2.load_state_dict(model1.state_dict())

        # Create MLL objects
        mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
        mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)

        # 3. Run fit_linear_mll
        # fit_linear_mll requires the model to be in train mode
        model1.train()
        likelihood1.train()
        fit_linear_mll(mll1)

        # 4. Run botorch fit
        model2.train()
        likelihood2.train()
        fit_gpytorch_mll(mll2)

        # 5. Compare Results
        model1.eval()
        model2.eval()
        likelihood1.eval()
        likelihood2.eval()

        # A. Compare MLL values on training data
        with torch.no_grad():
            output1 = model1(self.X)
            loss1 = -mll1(output1, self.Y.squeeze())

            output2 = model2(self.X)
            loss2 = -mll2(output2, self.Y.squeeze())

        logger.info(f"Fit Linear MLL Loss: {loss1.item():.6f}")
        logger.info(f"Botorch MLL Loss:    {loss2.item():.6f}")

        # Both use L-BFGS-B (scipy), so they should be very close.
        self.assertTrue(
            torch.allclose(loss1, loss2, atol=1e-3, rtol=1e-3),
            f"MLL values differ significantly: {loss1.item()} vs {loss2.item()}",
        )

        # Compare Noise
        noise1 = model1.likelihood.noise
        noise2 = model2.likelihood.noise
        logger.info(f"Noise (Linear):  {noise1.item():.6f}")
        logger.info(f"Noise (Botorch): {noise2.item():.6f}")

        # Compare Lengthscale
        ls1 = model1.covar_module.lengthscale
        ls2 = model2.covar_module.lengthscale
        logger.info(f"Lengthscale (Linear):  {ls1.item():.6f}")
        logger.info(f"Lengthscale (Botorch): {ls2.item():.6f}")

        # Compare Mean Constant
        mean1 = model1.mean_module.constant
        mean2 = model2.mean_module.constant
        logger.info(f"Mean (Linear):  {mean1.item():.6f}")
        logger.info(f"Mean (Botorch): {mean2.item():.6f}")

    def test_compare_fit_linear_mll_vs_botorch_n_less_than_d(self):
        """Compare fit_linear_mll vs botorch.fit_gpytorch_mll for n < d."""
        # Generate synthetic data: y = w^T x + noise
        self.n = 510
        self.d = 4000
        self.X = torch.rand(self.n, self.d)
        self.weights = torch.randn(self.d, 1)
        self.Y = self.X @ self.weights + 0.05 * torch.randn(self.n, 1)
        self.bounds = (0.0, 1.0)

        # 1. Setup Model for fit_linear_mll
        # We use LinearKernel directly as covar_module. SingleTaskGP will use it as is.
        # We initialize with a specific lengthscale prior/constraint via the class defaults,
        # but we'll ensure they start at the same point.
        likelihood1 = GaussianLikelihood()
        covar1 = LinearKernel(ard_num_dims=None, bounds=self.bounds)
        model1 = SingleTaskGP(self.X, self.Y, likelihood=likelihood1, covar_module=covar1)

        # 2. Setup Model for botorch fit (Baseline)
        likelihood2 = GaussianLikelihood()
        covar2 = LinearKernel(ard_num_dims=None, bounds=self.bounds)
        model2 = SingleTaskGP(self.X, self.Y, likelihood=likelihood2, covar_module=covar2)

        # Ensure both models start with identical parameters
        # model2.load_state_dict(model1.state_dict())

        # Create MLL objects
        mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
        mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)

        # 3. Run fit_linear_mll
        # fit_linear_mll requires the model to be in train mode
        model1.train()
        likelihood1.train()
        fit_linear_mll(mll1)

        # 4. Run botorch fit
        model2.train()
        likelihood2.train()
        fit_gpytorch_mll(mll2)

        # 5. Compare Results
        model1.eval()
        model2.eval()
        likelihood1.eval()
        likelihood2.eval()

        # A. Compare MLL values on training data
        with torch.no_grad():
            output1 = model1.forward(self.X)
            loss1 = -mll1(output1, model1.train_targets)

            output2 = model2.forward(self.X)
            loss2 = -mll2(output2, model2.train_targets)

        logger.info(f"\nFit Linear MLL Loss: {loss1.item():.6f}")
        logger.info(f"Botorch MLL Loss:    {loss2.item():.6f}")

        # Both use L-BFGS-B (scipy), so they should be very close.
        self.assertTrue(
            torch.allclose(loss1, loss2, atol=1e-3, rtol=1e-3) or loss1 < loss2,
            f"MLL values differ significantly: {loss1.item()} vs {loss2.item()}",
        )

        # Compare Noise
        noise1 = model1.likelihood.noise
        noise2 = model2.likelihood.noise
        logger.info(f"Noise (Linear):  {noise1.item():.6f}")
        logger.info(f"Noise (Botorch): {noise2.item():.6f}")

        # Compare Lengthscale
        ls1 = model1.covar_module.lengthscale
        ls2 = model2.covar_module.lengthscale
        logger.info(f"Lengthscale (Linear):  {ls1.item():.6f}")
        logger.info(f"Lengthscale (Botorch): {ls2.item():.6f}")

        # Compare Mean Constant
        mean1 = model1.mean_module.constant
        mean2 = model2.mean_module.constant
        logger.info(f"Mean (Linear):  {mean1.item():.6f}")
        logger.info(f"Mean (Botorch): {mean2.item():.6f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
