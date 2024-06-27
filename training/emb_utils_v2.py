"""
Add energy-based model for regression (ECCV 2020)
"""


import torch
import torch.distributions as dist

import math
import numpy as np

import pdb


num_samples = 1024

stds = torch.zeros((1, 2))
# stds[0, 0] = 0.1
# stds[0, 1] = 0.8

stds[0, 0] = 0.1    # 0.1    # 1
stds[0, 1] = 1.5    # 0.5    # 2


# def sample_gmm(y, stds, num_samples):
#     num_queries, num_dims = y.shape[1:]
#     L = stds.numel()
#
#     # Initialize a list for samples from each Gaussian
#     samples_list = []
#     log_probs_list = []
#
#     # Create and sample from each Gaussian distribution
#     for sigma_l in stds.view(-1):
#         mu_l = torch.zeros((num_dims)).to(y.device)
#         I_l = torch.eye(num_dims).to(y.device) * sigma_l ** 2
#         gaussian = dist.multivariate_normal.MultivariateNormal(mu_l, I_l)
#         samples_l = gaussian.sample((num_samples, num_queries))
#         log_probs_list.append(gaussian.log_prob(samples_l))
#         samples_l = samples_l.unsqueeze(0) + y.unsqueeze(1)
#         samples_list.append(samples_l)
#
#     # Stack all samples and log probabilities along a new dimension
#     all_samples = torch.stack(samples_list)
#     all_log_probs = torch.stack(log_probs_list)
#
#     # Average the samples from all distributions
#     mixed_samples = torch.mean(all_samples, dim=0)  # (b, num_samples, ep_len, act_dim)
#
#     # Calculate the log probabilities using logsumexp for numerical stability
#     log_probs_mixed = torch.logsumexp(all_log_probs, dim=0) - torch.log(torch.tensor(float(L)))
#
#     log_probs_mixed = log_probs_mixed.sum(-1)
#
#     log_probs_mixed_max = log_probs_mixed.max().item()
#     log_probs_mixed_min = log_probs_mixed.min().item()
#     log_probs_mixed = (log_probs_mixed - log_probs_mixed_max) / (log_probs_mixed_max - log_probs_mixed_min)
#
#     # # Convert log probabilities back to probabilities
#     # probs_mixed = torch.exp(log_probs_mixed)  # (num_samples,)
#
#     return mixed_samples, log_probs_mixed


def sample_gmm(y, stds, num_samples, sample_mode='random', ratio=0.0):
    num_queries, num_dims = y.shape[1:]
    L = stds.numel()

    # Initialize a list for samples from each Gaussian
    samples_list = []
    log_probs_list = []

    if sample_mode == 'random':
        # Sample indices from the uniform distribution over components
        component_indices = torch.randint(low=0, high=L, size=((num_samples, num_queries))).to(y.device)

        for sigma_l in stds.view(-1):
            mu_l = torch.zeros((num_dims)).to(y.device)
            I_l = torch.eye(num_dims).to(y.device) * sigma_l ** 2
            gaussian = dist.multivariate_normal.MultivariateNormal(mu_l, I_l)
            samples_l = gaussian.sample((num_samples, num_queries))
            log_probs_l = gaussian.log_prob(samples_l)
            log_probs_list.append(log_probs_l)
            samples_l = samples_l.unsqueeze(0) + y.unsqueeze(1)
            samples_list.append(samples_l)

        for l in range(L):
            samples_list[l] = samples_list[l] * (component_indices == l)[None, ..., None]
            log_probs_list[l] = log_probs_list[l] * (component_indices == l)

    elif sample_mode in ['inc', 'dec']:
        if sample_mode == 'inc':
            std_list = np.linspace(stds[0][0], stds[0][1], 5)
        else:
            std_list = np.linspace(stds[0][1], stds[0][0], 5)

        sigma_l = std_list[int(ratio * len(std_list) - 1e-6)]
        mu_l = torch.zeros((num_dims)).to(y.device)
        I_l = torch.eye(num_dims).to(y.device) * sigma_l ** 2
        gaussian = dist.multivariate_normal.MultivariateNormal(mu_l, I_l)
        samples_l = gaussian.sample((num_samples, num_queries))
        log_probs_l = gaussian.log_prob(samples_l)
        log_probs_list.append(log_probs_l)
        samples_l = samples_l.unsqueeze(0) + y.unsqueeze(1)
        samples_list.append(samples_l)
    else:
        raise ValueError(f'Invalid sample mode: {sample_mode}.')

    all_samples = torch.stack(samples_list).sum(0)
    all_log_probs = torch.stack(log_probs_list).sum(0)

    all_log_probs = all_log_probs.sum(-1)

    log_probs_max = all_log_probs.max().item()
    log_probs_min = all_log_probs.min().item()
    all_log_probs = (all_log_probs - log_probs_max) / (log_probs_max - log_probs_min)

    return all_samples, all_log_probs


# def sample_gmm(y, stds, num_samples, sample_mode='random', ratio=0.0):
#     num_queries, num_dims = y.shape[1:]
#     L = stds.numel()
#
#     # # Creating Gaussian distributions for each component
#     # gaussians = [dist.MultivariateNormal(mu_l, torch.eye(num_dims).to(y.device) * sigma_l ** 2) for sigma_l in stds.view(-1)]
#
#     # Initialize a list for samples from each Gaussian
#     samples_list = []
#     log_probs_list = []
#
#     # Sample indices from the uniform distribution over components
#     component_indices = torch.randint(low=0, high=L, size=((num_samples, num_queries))).to(y.device)
#
#     for sigma_l in stds.view(-1):
#         mu_l = torch.zeros((num_dims)).to(y.device)
#         I_l = torch.eye(num_dims).to(y.device) * sigma_l ** 2
#         gaussian = dist.multivariate_normal.MultivariateNormal(mu_l, I_l)
#         samples_l = gaussian.sample((num_samples, num_queries))
#         log_probs_l = gaussian.log_prob(samples_l)
#         log_probs_list.append(log_probs_l)
#         samples_l = samples_l.unsqueeze(0) + y.unsqueeze(1)
#         samples_list.append(samples_l)
#
#     for l in range(L):
#         samples_list[l] = samples_list[l] * (component_indices == l)[None, ..., None]
#         log_probs_list[l] = log_probs_list[l] * (component_indices == l)
#
#     all_samples = torch.stack(samples_list).sum(0)
#     all_log_probs = torch.stack(log_probs_list).sum(0)
#
#     all_log_probs = all_log_probs.sum(-1)
#
#     log_probs_max = all_log_probs.max().item()
#     log_probs_min = all_log_probs.min().item()
#     all_log_probs = (all_log_probs - log_probs_max) / (log_probs_max - log_probs_min)
#
#     return all_samples, all_log_probs


# def gauss_density_centered(x, std):
#     return torch.exp(-0.5 * (x / std) ** 2) / (math.sqrt(2 * math.pi) * std)
#
#
# def gmm_density_centered(x, std):
#     """
#     Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
#     """
#     pdb.set_trace()
#     if x.dim() == std.dim() - 1:
#         x = x.unsqueeze(-1)
#     elif not (x.dim() == std.dim() and x.shape[-1] == 1):
#         raise ValueError('Last dimension must be the gmm stds.')
#
#     return gauss_density_centered(x, std).prod(-2).mean(-1)



# def sample_gmm(y, stds, num_samples=1):
#     num_dims = np.prod(y.shape[1:])
#     num_components = stds.shape[-1]
#
#     stds = stds.repeat(num_dims, 1).view(1, num_dims, num_components)
#
#     # Sample component ids
#     k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
#     std_samp = stds[0, :, k].t()    # (num_samples, num_dims)
#     pdb.set_trace()
#     # Sample
#     x_centered = std_samp * torch.randn(num_samples, num_dims)
#     prob_dens = gmm_density_centered(x_centered, stds)
#     pdb.set_trace()
#     prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), stds)
#
#     return x_centered, prob_dens, prob_dens_zero