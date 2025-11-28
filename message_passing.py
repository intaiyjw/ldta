#!/usr/bin/python3
# -*- coding:utf-8 -*-
# 
# message_passing.py: given the expectation of sufficient statistics of a
# hyper-Dirichlet distribution, recover its parameter

import torch
from torch.special import digamma, polygamma
from typing import Callable

from dists import Dirichlet as Dir, \
                  Beta_Liouville as BL, \
                  Generalized_Dirichlet as GDir
from tensor_mat import tense_array_torch as expand_tensor

def inv_digamma(y, max_iter=50, tol=1e-10):
    y = torch.as_tensor(y)

    x = torch.where(
        y >= -2.22,
        torch.exp(y) + 0.5,
        -1.0 / (y - digamma(torch.tensor(1.0)))
    )

    i = 0
    delta = torch.full_like(x, float('inf'))

    while i < max_iter and torch.max(torch.abs(delta)) >= tol:
        delta = (digamma(x) - y) / polygamma(1, x)
        x = x - delta
        i += 1

    return x

def mean_u2alpha(mean_u: torch.Tensor, tol=1e-6):
    """
    Fixed point iteration
    Convert mean_u to alpha for a batch of regular Dirichlet
    Given a group of nonlinear equations:
        alpha_1 + alpha_2 + ... + alpha_K = alpha_0 
        psi(alpha_1) - psi(alpha_0) = mean_u_1
        psi(alpha_2) - psi(alpha_0) = mean_u_2
        ...
        psi(alpha_K) - psi(alpha_0) = mean_u_D
    retore alpha_1, alpha_2, ..., alpha_K

    Input:
    - mean_u: (..., K), the lower dimension is expections of sufficient statistics
    Returns:
    - alpha: (..., K), the recovered para 
    """
    # Initialization
    K = mean_u.shape[-1]
    alpha0_approx = K * 1.0
    psi_alpha0_approx = digamma(torch.tensor(alpha0_approx, 
                                             dtype=mean_u.dtype, 
                                             device=mean_u.device))
    alpha = inv_digamma(mean_u + psi_alpha0_approx)

    diff = float('inf')

    while diff > tol:
        alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)  # (..., 1)
        psi_alpha_0 = digamma(alpha_0)                    # (..., 1)
        alpha_new = inv_digamma(mean_u + psi_alpha_0)     # (..., K)

        diff = torch.max(torch.abs(alpha_new - alpha)).item()
        alpha = alpha_new

    return alpha


def mean_u2alpha_newton_original(mean_u: torch.Tensor, tol=1e-10):
    """
    Newton's method with fast Hessian inversion for Dirichlet
    Sherman-Morrison inversion
    double precision float is needed
    """
    alpha = torch.full_like(mean_u, fill_value=1.0)  # (..., K)

    for i in range(100):
        alpha_0 = alpha.sum(dim=-1, keepdim=True)  # (..., 1)
        d = digamma(alpha) - digamma(alpha_0) - mean_u  # (..., K)
        # if (d < tol).all():  !!! d could be negative, and exit iter early than expected
        # if (i > 30) and (d < tol).all():
        if (d.abs() < tol).all():
            break

        pi = 1.0 / (polygamma(1, alpha) + 1e-10)  # (..., K)
        sigma = pi.sum(dim=-1, keepdim=True) - 1.0 / (polygamma(1, alpha_0) + 1e-10)  # (..., 1)

        scalar_coeff = (pi * d).sum(dim=-1, keepdim=True) / (sigma + 1e-10)  # (..., 1)
        delta = (d - scalar_coeff) * pi  # (..., K)

        alpha -= delta  # (..., K)
        alpha = torch.clamp(alpha, min=1e-8)

    else:
        print("Warning: Newton's method reached max_iter without converging.")

    return alpha

def mean_u2alpha_newton(mean_u: torch.Tensor, tol=1e-8, max_iter=100):
    """
    Recover Dirichlet alpha from expected sufficient statistics using Newton's method.
    Uses Sherman-Morrison trick to avoid inverting the Hessian explicitly.

    Args:
        mean_u (Tensor): (..., K) tensor of E_q[log(theta_k)] - E_q[log(sum(theta))]
        tol (float): Tolerance for convergence (default: 1e-8).
        max_iter (int): Maximum number of Newton iterations (default: 100).

    Returns:
        alpha (Tensor): (..., K) estimated Dirichlet parameters.
    """
    # Ensure double precision and clean initialization
    # mean_u = mean_u.to(dtype=torch.float64)
    # alpha = torch.full_like(mean_u, fill_value=1.0, dtype=torch.float64)  # (..., K)
    alpha = torch.full_like(mean_u, fill_value=1.0)  # (..., K)

    # Initial clamp to ensure stability of special functions
    # alpha = torch.clamp(alpha, min=1e-8)

    for i in range(max_iter):
        alpha_0 = alpha.sum(dim=-1, keepdim=True)  # (..., 1)
        # alpha_0 = torch.clamp(alpha_0, min=1e-10)  # Protect special functions

        # Gradient
        grad = digamma(alpha) - digamma(alpha_0) - mean_u  # (..., K)

        # Convergence check
        if grad.abs().max() < tol:
            break

        # Hessian inverse via Sherman-Morrison
        psi1_alpha = polygamma(1, alpha)  # (..., K)
        psi1_alpha = torch.clamp(psi1_alpha, min=1e-10)
        pi = 1.0 / psi1_alpha  # (..., K)

        psi1_alpha0 = polygamma(1, alpha_0)  # (..., 1)
        psi1_alpha0 = torch.clamp(psi1_alpha0, min=1e-10)
        sigma = pi.sum(dim=-1, keepdim=True) - 1.0 / psi1_alpha0  # (..., 1)

        # Newton update using low-rank inverse
        scalar_coeff = (pi * grad).sum(dim=-1, keepdim=True) / sigma  # (..., 1)
        delta = pi * (grad - scalar_coeff)  # (..., K)

        alpha = alpha - delta
        alpha = torch.clamp(alpha, min=1e-8)  # maintain numerical stability

    else:
        print("Warning: Newton's method reached max_iter without converging.")

    return alpha


def mean_u2BL_para(mean_u: torch.Tensor, tol=1e-8):
    """
    Convert mean_u to para for a batch of Beta_Liouville
    Input:
    - mean_u: (..., K+1)
    Returns:
    - para: (..., K+1)
    """
    para_1 = mean_u2alpha_newton(mean_u[..., :-2], tol)  # (..., K-1)
    para_2 = mean_u2alpha_newton(mean_u[..., -2:], tol)  # (..., 2)
    return torch.cat((para_1, para_2), dim=-1)  # (..., K+1)

def mean_u2GDir_para(mean_u: torch.Tensor, tol=1e-8):
    """
    Convert mean_u to para for a batch of Generalized Dirichlet
    Input:
    - mean_u: (..., 2, K-1)
    Returns:
    - para: (..., 2, K-1)
    """
    mean_u_permuted = torch.einsum("...ij -> ...ji", mean_u)
    return torch.einsum("...ji -> ...ij", 
                        mean_u2alpha_newton(mean_u_permuted, tol))

def mean_u2dist(mean_u: torch.Tensor, dist_cls: Callable=Dir, tol=1e-8):
    if dist_cls == BL:
        para = mean_u2BL_para(mean_u, tol)
    elif dist_cls == GDir:
        para = mean_u2GDir_para(mean_u, tol)
    else:
        para = mean_u2alpha_newton(mean_u, tol)

    return dist_cls(para)

def main():
    pass

if __name__ == '__main__':
    main()
