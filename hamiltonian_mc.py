# -*- coding: utf-8 -*-

"""
Implementation of Hamiltonian Monte Carlo (HMC) algorithm in PyTorch that uses automatic differentiation and calculates
multiple chains in parallel with support of infinite log prob values.
Unlike some popular packages (e.g., Pyro), achieves faster parallelization by batched execution.
For a formal description of the HMC algorithm, see, for instance,
[1] `MCMC using Hamiltonian dynamics <https://arxiv.org/pdf/1206.1901.pdf>`, Radford M. Neal
"""

from typing import Tuple, List, Callable

import torch
from torch import Tensor

from tqdm import trange


def batched_hamiltonian_mc_step(
        log_prob_func: Callable[[Tensor], Tensor],
        params_init: Tensor,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a Hamiltonian Monte Carlo step in a batch.

    Args:
        log_prob_func (callable): Log probability function of the target distribution that supports batched execution.
        params_init (Tensor): Initial parameters with shape (num_chains, dim) or (dim, ) for the HMC step.
        num_steps_per_sample (int, optional): Number of leapfrog steps per sample. Defaults to 10.
        step_size (float, optional): Step size for the leapfrog integration. Defaults to 0.1.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing the updated parameters with shape (num_chains, dim)
                               and a boolean tensor with shape (num_chains, ) indicating
                               whether each sample was accepted or not.
    """

    assert params_init.dim() in (1, 2), f"Wrong dimensionality of the initial params. " \
                                        f"Expected dim == 1 or dim == 2, got {params_init.dim()}"

    # if 1-dim params for a single chain are provided, extend for 2-dim
    params_init = torch.atleast_2d(params_init)

    # clone params to perform further in-place operations
    params = params_init.clone()

    # generate auxiliary momentum values
    momentum = torch.randn_like(params)

    # calculate hamiltonian for the initial parameters (momentum will be changed in-place afterwards)
    ham = hamiltonian(params, momentum, log_prob_func)

    # perform batched leapfrog step:
    params, momentum, finite_indices = batched_leapfrog(
        params, momentum, log_prob_func,
        steps=num_steps_per_sample, step_size=step_size,
    )

    # calculate hamiltonian for the new parameters:
    new_ham = torch.ones_like(ham) * float('inf')
    new_ham[finite_indices] = hamiltonian(
        params[finite_indices], momentum[finite_indices], log_prob_func
    )

    # accept / reject proposals via a Metropolis update
    rho = torch.clamp_max(ham - new_ham, 0.)
    rejection_condition = rho < torch.log(torch.rand_like(rho))
    num_rejected = rejection_condition.sum().item()
    if num_rejected > 0:
        # return initial parameters instead of rejected proposals
        params[rejection_condition] = params_init[rejection_condition]

    accepted = ~rejection_condition

    return params, accepted


def hamiltonian(params: Tensor, momentum: Tensor, log_prob_func: Callable[[Tensor], Tensor]) -> Tensor:
    """
    Calculate the Hamiltonian for a given set of parameters and momentum.

    Args:
        params (Tensor): Parameters with shape (num_chains, dim).
        momentum (Tensor): Momentum with shape (num_chains, dim) associated with the parameters.
        log_prob_func (callable): Log probability function of the target distribution.

    Returns:
        Tensor: Hamiltonian values with shape (num_chains, ) for the given parameters and momentum.
    """
    return -log_prob_func(params) + 0.5 * torch.sum(momentum * momentum, dim=-1)


def batched_leapfrog(
        params: Tensor, momentum: Tensor,
        log_prob_func: Callable[[Tensor], Tensor],
        steps: int = 10,
        step_size: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform the leapfrog integration step in the Hamiltonian Monte Carlo algorithm for a batch of parameters.

    Args:
        params (Tensor): Parameters with shape (num_chains, dim).
        momentum (Tensor): Momentum with shape (num_chains, dim) associated with the parameters.
        log_prob_func (callable): Log probability function of the target distribution.
        steps (int, optional): Number of leapfrog steps. Defaults to 10.
        step_size (float, optional): Step size for the leapfrog integration. Defaults to 0.1.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Tuple containing the updated parameters, momentum, and a boolean
                                       tensor indicating whether each sample was finite or not.
    """

    grad, finite_indices = _get_batched_params_grad(params, log_prob_func)
    momentum[finite_indices] += 0.5 * step_size * grad

    for n in range(steps):
        if finite_indices.sum().item() == 0:
            # all the proposals are outside the prior distribution, so we stop here
            break

        params[finite_indices] += step_size * momentum[finite_indices]

        grad, finite_indices = _get_batched_params_grad(params, log_prob_func, finite_indices)

        momentum[finite_indices] += step_size * grad

    momentum[finite_indices] -= 0.5 * step_size * grad

    return params, momentum, finite_indices


def _get_batched_params_grad(p, func, finite_indices=None):
    """
    Calculate the gradient of the log probability function for a batch of parameters.

    Args:
        p (Tensor): Parameters with shape (num_chains, dim).
        func (callable): Log probability function of the target distribution.
        finite_indices (Tensor, optional): Boolean tensor indicating which samples have finite log prob.
                                           Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
                                1. Tensor with gradients of the log probability function
                                   w.r.t. parameters with finite log prob. Shape = (m, dim), m <= num_chains
                                2. Boolean tensor finite_indices indicating which samples have finite log prob,
                                   such that finite_indices.sum() == m.
    """

    if finite_indices is not None:
        p = p[finite_indices]

    p.detach_().requires_grad_()
    log_probs = func(p)
    new_finite_indices = torch.isfinite(log_probs)
    if finite_indices is not None:
        finite_indices[finite_indices.clone()] = new_finite_indices
    else:
        finite_indices = new_finite_indices
    grad = torch.autograd.grad(log_probs.sum(), p, allow_unused=True)[0][new_finite_indices]
    p.detach_()
    assert grad.shape[0] == finite_indices.sum().item(), f"{grad.shape[0]} != {finite_indices.sum().item()}"
    return grad, finite_indices


def run_hmc(
        log_prob_func: Callable[[Tensor], Tensor],
        init_params: Tensor,
        num_steps: int,
        burn_in: int = -1,
        thinning: int = 1,
        num_steps_per_sample: int = 10,
        step_size: float = 0.1,
        disable_tqdm: bool = False,
        storage_device: str = 'cpu',
) -> Tensor:
    """
    Run Hamiltonian Monte Carlo sampler.

    Args:
        log_prob_func (callable): Log probability function of the target distribution.
        init_params (Tensor): Initial parameters with shape (num_chains, dim) or (dim, ) for the HMC step.
        num_steps (int): Number of MCMC steps.
        burn_in (int): Length of burn-in period when samples are not stored.
        thinning (int): Thinning step. Defaults to 1.
        num_steps_per_sample (int):  Number of leapfrog steps per sample. Defaults to 10.
        step_size (int): Step size for the leapfrog integration. Defaults to 0.1.
        disable_tqdm (bool): Whether to disable tqdm progress bar. Defaults to False.
        storage_device (str): Device where to store sampled parameters. Defaults to 'cpu'.

    Returns:
        Tensor with shape (n, num_chains, dim),
               where n = (num_steps - burn_in) // thinning is the number of saved MCMC steps.
    """

    if burn_in >= num_steps:
        raise ValueError(f"Burn-in period {burn_in} must be smaller than the total number of steps {num_steps}.")
    if thinning > (num_steps - burn_in):
        raise ValueError(f"Thinning is too large; no samples will be saved.")

    mcmc_samples: List[Tensor] = []
    params: Tensor = init_params

    pbar = trange(num_steps, disable=disable_tqdm)
    _sum_mean_accepted = 0

    for i in pbar:
        params, accepted = batched_hamiltonian_mc_step(
            log_prob_func, params,
            num_steps_per_sample=num_steps_per_sample,
            step_size=step_size,
        )
        _sum_mean_accepted += accepted.to(float).mean()
        pbar.set_description(
            f'Average accepted ratio = {(_sum_mean_accepted / (i + 1)) * 100:.2f} %'
        )
        if i > burn_in and i % thinning == 0:
            mcmc_samples.append(params.to(storage_device))

    return torch.stack(mcmc_samples)
