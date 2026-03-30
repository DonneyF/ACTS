"""
Main module for running Bayesian optimization loops.
"""

import functools
import gc
import gzip
import logging
import math
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from queue import Queue
from time import perf_counter
from typing import Any, Callable

import botorch
import gpytorch
import hydra
import linear_operator
import numpy as np
import pandas as pd
import torch
import tqdm as tqdm
import wandb
from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms import Standardize, Warp
from botorch.test_functions.base import BaseTestProblem
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from jaxtyping import Float, Int64
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from botorch.utils import standardize

from src.priors import LogNormalPrior


DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}

MEANS = {
    "constant": lambda: gpytorch.means.ConstantMean(),
    "zero": lambda: gpytorch.means.ZeroMean(),
}

LIKELIHOODS = {
    "gaussian": lambda device: gpytorch.likelihoods.GaussianLikelihood(),
    "gaussian_dsp": lambda device: gpytorch.likelihoods.GaussianLikelihood(
        noise_prior := LogNormalPrior(loc=-4.0, scale=1.0, device=device),
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    ),
    "fixed_0.1": lambda device: (
        likelihood := gpytorch.likelihoods.GaussianLikelihood(),
        setattr(likelihood, "noise", 0.1),
        likelihood.raw_noise.requires_grad_(False),
    )[0],
}

def _build_scale_kernel(base_kernel, outputscale_type, outputscale_param):
    scale_kernel = ScaleKernel(base_kernel)
    match outputscale_type:
        case "fixed":
            scale_kernel.outputscale = outputscale_param
            scale_kernel.raw_outputscale.requires_grad_(False)
        case "learn":
            pass
        case _:
            raise NotImplementedError

    return scale_kernel


def initialize_model(
    X_train: Float[Tensor, "n d"],
    Y_train: Float[Tensor, "n 1"],
    model_config: DictConfig,
    bounds: Float[Tensor, "2 d"],
) -> botorch.models.SingleTaskGP:
    """
    Initialize the model for Bayesian optimization.

    :param X_train: Training input points.
    :param Y_train: Training output points.
    :param model_config: Model configuration.
    :return: Initialized model.
    """
    d = X_train.size(-1)
    base_kernel = hydra.utils.instantiate(model_config.kernel.fn, d=d).to(device=X_train.device)

    match model_config.outcome_transform:
        case "standardize":
            outcome_tf = Standardize(m=Y_train.shape[-1], batch_shape=X_train.shape[:-2])
        case None:
            outcome_tf = None
        case _:
            raise ValueError(f"Unknown outcome transform: {model_config.outcome_transform}")

    return hydra.utils.instantiate(
        model_config.gp.fn,
        train_X=X_train,
        train_Y=Y_train,
        mean_module=MEANS[model_config.mean](),
        covar_module=_build_scale_kernel(base_kernel, model_config.outputscale.type, model_config.outputscale.param),
        likelihood=LIKELIHOODS[model_config.likelihood](X_train.device),
        outcome_transform=outcome_tf,
    )


def evaluate_y(
    test_function: BaseTestProblem,
    X: Float[Tensor, "... d"],
    unnormalize: bool = True,
) -> tuple[Float[Tensor, "..."], dict | None]:
    r"""
    Evaluate a test function at a given (batch of) input(s) :math:`X`.

    :param test_function: The test function to evaluate.
    :param X: The input(s) at which to evaluate the function.
    :param unnormalize: Whether to unnormalize the inputs to the test function's bounds. If False, evaluates the
                        function ignoring the bounds.
    """

    Y = test_function(botorch.utils.transforms.unnormalize(X, test_function.bounds))
    return Y


def save_ckpt(ckpt: dict, path: Path, compress: bool) -> None:
    """
    Save a checkpoint to disk.

    :param ckpt: Checkpoint to save.
    :param path: Path to save the checkpoint.
    :param compress: Whether to compress the checkpoint.
    """
    ckpt_file = path / Path("checkpoint.pt")
    dest_path = ckpt_file.with_name(ckpt_file.name + ".gz") if compress else ckpt_file
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    opener = gzip.open(tmp_path, "wb") if compress else open(tmp_path, "wb")
    with opener as f:
        torch.save(ckpt, f)
    tmp_path.replace(dest_path)


def load_ckpt(path: Path, map_location: torch.device, compress: bool) -> dict | None:
    """
    Loads a checkpoint file from output directory

    :param path: Path to checkpoint folder
    :param map_location: Which device to load tensors
    :param compress: True if checkpoint is compressed
    :return: Checkpoint or ``None`` if the checkpoint is not found.
    """
    ckpt_file = path / Path("checkpoint.pt")
    try:
        if compress:
            with gzip.open(ckpt_file.with_name(ckpt_file.name + ".gz"), "rb") as f:
                return torch.load(f, map_location=map_location)
        else:
            with open(ckpt_file, "rb") as f:
                return torch.load(f, map_location=map_location)
    except FileNotFoundError:
        return None


def fit_model(model, config: DictConfig) -> None:
    """
    Fits the GP to the training data.

    :param model: The model to fit.
    :param config: Model configuration.
    """
    model.train()
    if isinstance(model, (SingleTaskGP, SaasFullyBayesianSingleTaskGP)):
        num_data = len(model.train_targets)
    elif isinstance(model, SingleTaskVariationalGP):
        num_data = len(model.model.train_targets)
    else:
        raise NotImplementedError

    base_kernel = model.covar_module.base_kernel

    def fit_model_(mll):
        try:
            botorch.fit.fit_gpytorch_mll(mll, optimizer=botorch.fit.fit_gpytorch_mll_scipy)
        except (botorch.exceptions.ModelFittingError, linear_operator.utils.errors.NotPSDError):
            # Failed to fit using scipy, fall back to pytorch (uses Adam).
            botorch.fit.fit_gpytorch_mll(mll, optimizer=botorch.fit.fit_gpytorch_mll_torch)

    with ExitStack() as es:
        es.enter_context(gpytorch.settings.cholesky_max_tries(10))

        if config.model_fitting.log_prob and num_data >= config.model_fitting.log_prob_threshold:
            # Approximate MLL computation with conjugate gradients
            es.enter_context(gpytorch.settings.max_cholesky_size(0))
            es.enter_context(
                gpytorch.settings.fast_computations(log_prob=True, covar_root_decomposition=True, solves=True)
            )
        else:
            # Just use Cholesky
            es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
            es.enter_context(
                gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves=False)
            )

        # Fit the model
        if isinstance(model, (SingleTaskGP, SingleTaskVariationalGP)):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            fit_model_(mll)
        elif isinstance(model, SaasFullyBayesianSingleTaskGP):
            botorch.fit.fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=512,
                num_samples=256,
                thinning=16,
                disable_progbar=True,
                jit_compile=True,
            )


def run_optimization(config: DictConfig) -> None:
    r"""
    Main function to run a BO loop.

    :params config: Configuration object containing the parameters for the BO loop.
    """
    # Resolve configuration
    OmegaConf.resolve(config)
    logging.info("\n" + OmegaConf.to_yaml(config))
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name

    # Set seeds
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    # Dtype and device
    dtype = DTYPES[config.dtype]
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")

    # Set maximum GPU allocation based on the number of parallel tasks from Slurm launcher.
    if device.type == "cuda":
        tasks_per_node = hydra.core.hydra_config.HydraConfig.get().launcher.get("tasks_per_node", 1)
        torch.cuda.set_per_process_memory_fraction(1.0 / tasks_per_node)

    # Get benchmark
    test_function = hydra.utils.instantiate(config.benchmark.fn).to(dtype=dtype, device=device)
    symmetric_test_fn_bounds = torch.allclose(test_function.bounds[1], -test_function.bounds[0])
    max_unnormalized_radius = test_function.bounds[1, 0].item() if symmetric_test_fn_bounds else None

    # Lower and upper bound for model (normalized to be in [0, 1]^d hypercube)
    lb = torch.zeros(test_function.dim, dtype=dtype, device=device)
    ub = torch.ones(test_function.dim, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=-2)
    n_tot = config.benchmark.n_tot
    d = test_function.dim
    q = config.acquisition.q
    n_init = config.benchmark.n_init
    num_iter = math.ceil((n_tot - n_init) / q)
    trust_region = hydra.utils.instantiate(config.trust_region.fn, d, q, n_tot, n_init) if config.trust_region else None

    acqf = hydra.utils.instantiate(config.acquisition.fn)  # type: Acquisition

    # wandb run
    config_dict = OmegaConf.to_container(config)
    config_dict["job_name"] = job_name
    config_dict.pop("hydra", None)
    logging_config = config_dict.pop("logging", None)
    wandb_config = logging_config.get("wandb", None) or {"mode": "disabled"}

    # Restore run from previous checkpoint, or create a fresh run
    ckpt = (
        load_ckpt(output_dir, torch.device("cpu"), compress=config.checkpoint.compress)
        if config.checkpoint.enable
        else None
    )
    if ckpt:
        # Tensors for X,Y
        Xs = ckpt["Xs"] = ckpt["Xs"].to(device)
        Ys = ckpt["Ys"] = ckpt["Ys"].to(device)

        # Indices for state management
        itrs = ckpt["itrs"]
        itr = ckpt["itr"]
        n_evals = ckpt["n_evals"]
        n_tr = ckpt["n_tr"]

        # Lengthscales, noise, outputscales
        lengthscales = ckpt["lengthscales"]
        noises = ckpt["noises"]
        outputscales = ckpt["outputscales"]

        # W&B Resumption
        wandb_config["id"] = ckpt["wandb_run_id"]
        wandb_config["resume"] = "must"

        if n_evals == n_tot:
            logging.info("Nothing to do.")
            return 0
    else:
        # Construct tensors with x, y values
        Xs: Float[Tensor, "n_tot d"] = torch.empty((n_tot, d), dtype=dtype, device=device)
        Ys: Float[Tensor, "n_tot 1"] = torch.empty((n_tot, 1), dtype=dtype, device=device)

        # Get initial points
        Xs[:n_init, :] = hydra.utils.instantiate(
            config.initializer,
            n=n_init,
            d=d,
            bounds=bounds,
            test_function=test_function,
            seed=config.seed,
        ).to(Xs)
        Y_init = evaluate_y(test_function, Xs[:n_init])
        Ys[:n_init] = Y_init.unsqueeze(-1)

        # Indices for state/progress
        n_evals: Int64[Tensor, "1"] = torch.tensor(
            [n_init], dtype=torch.int64, device=torch.device("cpu")
        )  # Total number of evaluations (including TR restarts)
        n_tr: Int64[Tensor, "1"] = torch.tensor(
            [0], dtype=torch.int64, device=torch.device("cpu")
        )  # Index to manage leftpont of X/Y where TR has restarted
        itrs: Int64[Tensor, " n_tot"] = torch.zeros((n_tot,), dtype=torch.int64, device=torch.device("cpu"))
        itr: Int64[Tensor, "1"] = torch.tensor([0], dtype=torch.int64, device=torch.device("cpu"))

        # Tensors to track state indexed by iterations
        lengthscales: Float[Tensor, "num_iter+1 d"] = torch.empty(
            (num_iter + 1, d), dtype=dtype, device=torch.device("cpu")
        )
        noises: Float[Tensor, "num_iter+1 d"] = torch.empty((num_iter + 1, 1), dtype=dtype, device=torch.device("cpu"))
        outputscales: Float[Tensor, "num_iter+1 d"] = torch.empty(
            (num_iter + 1, 1), dtype=dtype, device=torch.device("cpu")
        )

        # Dictionary of torch tensors for checkpointing. W&B data is added later.
        ckpt = {
            "Xs": Xs,
            "Ys": Ys,
            "lengthscales": lengthscales,
            "noises": noises,
            "n_tr": n_tr,
            "itr": itr,
            "itrs": itrs,
            "n_evals": n_evals,
            "outputscales": outputscales,
        }

    # Compute current best observation
    idx_max: Int64[Tensor, ""] = Ys[n_tr:n_evals, 0].argmax()
    X_inc: Float[Tensor, " D"] = Xs[idx_max, :]
    y_max: float = Ys[idx_max, 0].item()
    iter_since_new_inc = 0 if n_evals.item() == n_init else (n_evals.item() - idx_max).item()

    with wandb.init(config=config_dict, **wandb_config) as run:
        # Logging of initial points
        if n_evals == n_init:
            run.log(data={"y_max": y_max, "y_curr": y_max}, step=n_init)

        ckpt["wandb_run_id"] = run.id

        # Construct iterator for BO loop
        tqdm_log_list = logging_config["tqdm"] if bool(logging_config.get("tqdm", False)) else []  # noqa: C416
        pbar = tqdm.tqdm(
            initial=n_evals.item(),
            total=n_tot,
            desc="BO loop",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            disable=(not len(tqdm_log_list)),
        )

        ###
        # BO loop
        ###
        wandb_log_queue = Queue()  # Unused if we are not checkpointing
        n = n_evals
        try:
            n_prev: int = n_evals.item()
            while n_prev < n_tot:
                if config.checkpoint.enable and itr % config.checkpoint.interval == 0:
                    save_ckpt(ckpt, output_dir, compress=config.checkpoint.compress)
                    while not wandb_log_queue.empty():
                        iter_stats, step = wandb_log_queue.get()
                        run.log(iter_stats, step=step)

                q = min(config.acquisition.q, n_tot - n_prev)
                n = n_prev + q

                X = Xs[n_tr:n_prev]
                Y = Ys[n_tr:n_prev]

                # Filter data if trust region is used
                if trust_region:
                    X, Y = trust_region.filter_data(X, Y)

                # Fit model
                with linear_operator.settings.verbose_linalg(config.logging.debug_linalg):
                    # ^^ Turn on this context manager to see the expensive linear algebra operations
                    # being performed under-the-hood
                    model = initialize_model(
                        X_train=X,
                        Y_train=standardize(Y),
                        model_config=config.model,
                        bounds=bounds,  # Bounds for data
                    )
                    fit_start = perf_counter()
                    fit_model(model, config)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    fit_end = perf_counter()
                    model.eval()

                    # Construct bounds, if we are using a trust region
                    if trust_region:
                        tr_bounds = trust_region.construct_trust_region(
                            model=model,
                            X=X,
                            Y=Y,
                        )
                        bounds_ = torch.stack(
                            [
                                torch.clamp(tr_bounds[0], min=bounds[0], max=bounds[1]),
                                torch.clamp(tr_bounds[1], min=bounds[0], max=bounds[1]),
                            ]
                        )
                    else:
                        bounds_ = bounds

                    # maximize acquisition function to get next Xs
                    acqf_start = perf_counter()
                    Xs[n_prev:n] = acqf.argmax(
                        model=model,
                        X=X,
                        Y=Y,
                        q=q,
                        bounds=bounds_,
                        options=config.acquisition.options,
                        max_unnormalized_radius=max_unnormalized_radius,
                        trust_region=trust_region,
                    )
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    acqf_end = perf_counter()

                # Observe next Ys
                eval_start = perf_counter()
                Y_next = evaluate_y(test_function, Xs[n_prev:n])
                Ys[n_prev:n] = Y_next.unsqueeze(-1)
                eval_end = perf_counter()

                with torch.no_grad():
                    # Compute distance from the incumbent
                    dist_from_inc = torch.norm(Xs[n_prev:n] - X_inc, dim=-1).min().item()

                    # See if we have a new best observation
                    y_curr = Ys[n_prev:n, 0].max().item()
                    if y_curr > y_max:
                        idx_max: Int64[Tensor, ""] = Ys[:n, 0].argmax()
                        X_inc: Float[Tensor, " D"] = Xs[idx_max, :]
                        y_max: float = Ys[idx_max, 0].item()
                        iter_since_new_inc = 0
                    else:
                        iter_since_new_inc += q

                    # Model hyperparameters
                    lengthscale = model.covar_module.base_kernel.lengthscale.squeeze().detach().cpu()
                    outputscale = model.covar_module.outputscale.detach().cpu()
                    noise = model.likelihood.noise.detach().cpu()

                    # Geometry of space and chosen point
                    _box_lengths = (bounds[1] - bounds[0]).detach().cpu().div(lengthscale)
                    _center = (bounds[1] + bounds[0]).detach().cpu().div(2.0)
                    expected_sq_norm = _box_lengths.square().sum(dim=-1).div(12.0).item()
                    rel_xcurr_sq_norm = (
                        Xs[n_prev:n].detach().cpu().sub(_center).div(lengthscale).square().sum(dim=-1).mean().item()
                    )
                    abs_curr_sq_norm = Xs[n_prev:n].detach().cpu().square().sum(dim=-1).mean().item()

                    # Update progress bar
                    iter_stats = OrderedDict(
                        y_max=y_max,
                        y_curr=y_curr,
                        iter_since_new_inc=iter_since_new_inc,
                        ls_min=lengthscale.min().item(),
                        ls_max=lengthscale.max().item(),
                        ls_mean=lengthscale.mean().item(),
                        os=outputscale.mean().item(),
                        noise=noise.item(),
                        dist_from_inc=dist_from_inc,
                        expected_sq_norm=expected_sq_norm,
                        rel_xcurr_sq_norm=rel_xcurr_sq_norm,
                        abs_curr_sq_norm=abs_curr_sq_norm,
                        fit_time=fit_end - fit_start,
                        acqf_time=acqf_end - acqf_start,
                        obj_eval_time=eval_end - eval_start,
                    )
                    if device.type == "cuda":
                        iter_stats["torch_mem_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                        torch.cuda.reset_peak_memory_stats()
                    pbar.set_postfix(**{stat: iter_stats.get(stat, None) for stat in tqdm_log_list})

                    # Log results to wandb
                    if config.checkpoint.enable:
                        wandb_log_queue.put((iter_stats, n))
                    else:
                        run.log(data=iter_stats, step=n)

                # Trust region update
                if trust_region is not None:
                    if trust_region.restart_triggered:
                        # Create new points
                        n_restart = n_init
                        if n_tot - n < n_restart and n_restart >= q:
                            # We are close to the end of optimization. Insufficient budget, do not restart.
                            break

                        X_init = hydra.utils.instantiate(
                            config.initializer,
                            n=n_init,
                            d=d,
                            bounds=bounds,
                            test_function=test_function,
                        ).to(Xs)
                        Y_init = evaluate_y(test_function, X_init).unsqueeze(-1)

                        Xs[n : n + n_restart] = X_init
                        Ys[n : n + n_restart] = Y_init

                        trust_region = hydra.utils.instantiate(config.trust_region.fn, d, q, n_tot, n_init)

                        # Ignore all previous observations due to restart.
                        n_tr = n

                        pbar.update(n_restart)
                    else:
                        n_restart = 0
                        trust_region.update_state(Xs[n_prev:n], Ys[n_prev:n])
                else:
                    n_restart = 0

                # Store index, GP hypers
                itrs[n_prev : n + n_restart] = itr
                lengthscales[itr] = lengthscale.detach().cpu()
                noises[itr] = noise.detach().cpu()
                outputscales[itr] = outputscale.detach().cpu()

                n_prev = n + n_restart
                n_evals.fill_(n_prev)
                itr += 1
                pbar.update(q)

                # Empty caches
                model._clear_cache()
        finally:
            # Checkpoint remaining
            if config.checkpoint.enable and n == n_tot:
                ckpt["wandb_run_id"] = run.id
                save_ckpt(ckpt, output_dir, compress=config.checkpoint.compress)
                while not wandb_log_queue.empty():
                    iter_stats, step = wandb_log_queue.get()
                    run.log(data=iter_stats, step=step)

            # Send wandb artifacts if we are at the end of optimization when checkpointing
            # or always when not checkpointing
            if (config.checkpoint.enable and n == n_tot) or not config.checkpoint.enable:
                data = {
                    "lengthscales": wandb.Table(dataframe=pd.DataFrame(lengthscales[:itr].cpu().numpy())),
                    "outputscales": wandb.Table(dataframe=pd.DataFrame(outputscales[:itr].cpu().numpy())),
                    "noises": wandb.Table(dataframe=pd.DataFrame(noises[:itr].cpu().numpy())),
                    "itrs": wandb.Table(dataframe=pd.DataFrame(itrs[:n].cpu().numpy())),
                    "Xs": wandb.Table(dataframe=pd.DataFrame(Xs[:n].cpu().numpy())),
                    "Ys": wandb.Table(dataframe=pd.DataFrame(Ys[:n].cpu().numpy())),
                }
                data = {key: val for key, val in data.items() if key in config.logging.wandb_artifacts}
                run.log(data, step=n + 1)  # n is reserved for last observation

        pbar.close()

    logging.info(f"Best observation: {y_max}")
