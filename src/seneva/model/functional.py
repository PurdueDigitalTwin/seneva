# Copyright 2023 (c) David Juanwu Lu and Purdue Digital Twin Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Functional components for the neural network model."""
import math
from typing import Optional

import torch

from seneva.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)


@torch.jit.script
def sinusoidal_pe(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the generalized sinusoidal positional encoding.

    Args:
        pos (torch.Tensor): Input position tensor.
        dim (int): The dimension of positional embeddings.

    Returns:
        torch.Tensor: Encoded positional embeddings.
    """
    assert pos.size(-1) == 2, "Invalid position tensor."
    div_term = torch.exp(
        -math.log(10000.0)
        / dim
        * torch.arange(0, dim, 2, device=pos.device, dtype=pos.dtype)
    )
    x_pos = 2 * math.pi * pos[..., 0:1] / div_term
    x_emb = torch.stack(
        [
            torch.sin(x_pos[..., 0::2]),
            torch.cos(x_pos[..., 0::2]),
        ],
        dim=-1,
    ).flatten(-2)
    y_pos = 2 * math.pi * pos[..., 1:2] / div_term
    y_emb = torch.stack(
        [
            torch.sin(y_pos[..., 0::2]),
            torch.cos(y_pos[..., 0::2]),
        ],
        dim=-1,
    ).flatten(-2)

    return torch.cat([x_emb, y_emb], dim=-1)


def compute_circular_iou(
    selected_samples: torch.Tensor,
    other_samples: torch.Tensor,
    selected_radius: float = 2.5,
    other_radius: Optional[float] = None,
) -> torch.Tensor:
    """Compute the IoU between two sets of samples regarding circular buffers.

    Args:
        selected_samples (torch.Tensor): Selected samples of shape
            :math:`[num_agents, num_samples, 2]`.
        other_samples (torch.Tensor): Other samples of shape
            :math:`[num_agents, num_samples, 2]`.
        selected_radius (float): Radius of the circular buffer for the selected
            samples in meters. Defaults to `2.5`.
        other_radius (Optional[float]): Radius of the circular buffer for the
            other samples in meters. If `None`, use `selected_radius`.
            Defaults to `None`.

    Returns:
        torch.Tensor: The IoU between the two sets of samples.
    """
    if other_radius is None:
        other_radius = selected_radius
    if selected_samples.ndim == other_samples.ndim - 1:
        selected_samples = selected_samples.unsqueeze(-2)

    assert selected_samples.ndim == 3 and selected_samples.size(-1) == 2
    assert other_samples.ndim == 3 and other_samples.size(-1) == 2

    d = torch.cdist(other_samples, selected_samples, p=2).squeeze(-1)
    y = 0.5 * torch.sqrt(
        (-d + selected_radius + other_radius)
        * (d + selected_radius - other_radius)
        * (d - selected_radius + other_radius)
        * (d + selected_radius + other_radius)
    )
    area_intersection = (
        selected_radius**2
        * torch.acos(
            (d.pow(2) + selected_radius**2 - other_radius**2)
            / (2 * d * selected_radius)
        )
        + other_radius**2
        * torch.acos(
            (d.pow(2) - selected_radius**2 + other_radius**2)
            / (2 * d * other_radius)
        )
        - y
    )
    area_union = (
        selected_radius**2 * math.pi
        + other_radius**2 * math.pi
        - area_intersection
    )
    iou = area_intersection / area_union

    # handle extreme cases: no intersection or full intersection
    iou[torch.isclose(d, torch.tensor(0.0))] = 1.0
    iou[d >= selected_radius + other_radius] = 0.0

    return iou


@torch.jit.script
def kl_divergence_categorical(
    log_p: torch.Tensor, log_q: torch.Tensor
) -> torch.Tensor:
    """Compute the KL[p||q] between two categorical distributions.

    .. math::
        D_{KL}(p||q) = \\sum_i p_i \\log \\frac{p_i}{q_i}

    Args:
        log_p (torch.Tensor): The log-probabilities of distribution :math:`p`.
        log_q (torch.Tensor): The log-probabilities of distribution :math:`q`.

    Returns:
        torch.Tensor: The scalr KL divergence :math:`KL[p||q]`.

    Raises:
        ValueError: If the shapes of the input tensors do not match or the
            input tensors are not valid probabilities.
    """
    # if not torch.allclose(log_p.exp().sum(-1), torch.tensor(1.0), atol=1e-3):
    # raise ValueError("The input tensor `p` is not a valid probability.")
    # if not torch.allclose(log_q.exp().sum(-1), torch.tensor(1.0), atol=1e-3):
    # raise ValueError("The input tensor `q` is not a valid probability.")
    if not log_p.shape[-1] == log_q.shape[-1]:
        raise ValueError("The shapes of the input tensors do not match.")

    return torch.sum(log_p.exp() * (log_p - log_q), dim=-1)


@torch.jit.script
def kl_divergence_diagonal_gaussians(
    p_mean: torch.Tensor,
    p_prec: torch.Tensor,
    q_mean: torch.Tensor,
    q_prec: torch.Tensor,
) -> torch.Tensor:
    """Compute the KL[p||q] between two diagonal Gaussian distributions:

    .. math::
        D_{KL}(p||q) = \\frac{1}{2} \\left( \\text{tr}(\\Sigma_q^{-1} \\Sigma_p) + (\\mu_p - \\mu_q)^T \\Sigma_q^{-1} (\\mu_p - \\mu_q) + \\log \\frac{\\det \\Sigma_q}{\\det \\Sigma_p} - d \\right),
        where :math:`d` is the dimension of the distribution.

    Args:
        p_mean (torch.Tensor): The mean of the distribution :math:`p`.
        p_prec (torch.Tensor): The precisions of the distribution :math:`p`.
        q_mean (torch.Tensor): The mean of the distribution :math:`q`.
        q_prec (torch.Tensor): The precisions of the distribution :math:`q`.

    Returns:
        torch.Tensor: The scalr KL divergence :math:`KL[p||q]`.

    Raises:
        ValueError: If the shapes of the input tensors do not match.
    """
    if p_mean.shape != q_mean.shape or p_prec.shape != q_prec.shape:
        raise ValueError("The shapes of the input tensors do not match.")

    diff = p_mean - q_mean
    mahalanobis = torch.square(diff).mul(q_prec).sum(dim=-1)
    trace = q_prec.div(p_prec).sum(dim=-1)
    logdet = torch.log(p_prec.div(q_prec)).sum(dim=-1)

    return 0.5 * (mahalanobis + trace + logdet - p_mean.shape[-1])


@torch.jit.script
def linear_gaussian_reconstruction_loss(
    y: torch.Tensor,
    x_vars: torch.Tensor,
    y_mean: torch.Tensor,
    weight: torch.Tensor,
    y_vars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the analytical reconstruction loss for a linear Gaussian model.

    Args:
        y (torch.Tensor): Ground-truth observation.
        x_vars (torch.Tensor): Variance vector of the latent variable.
        y_mean (torch.Tensor): Mean vector of the emission distribution.
        weight (torch.Tensor): Weight in the linear emission layer.
        y_vars (Optional[torch.Tensor]): Variance vector of the emission
            distribution. If `None`, ignore related terms. Defaults to `None`.

    Returns:
        torch.Tensor: The analytical reconstruction loss.
    """
    # sanity checks
    n, d = weight.shape
    if not y_mean.shape[-1] == n and x_vars.shape[-1] == d:
        raise ValueError("Shape mismatched!")

    if y_vars is None:
        # NOTE: when y_vars is not provided, treat it as a identity matrix
        # compute the mahalanobis term
        mahalanobis = torch.sum(torch.square(y - y_mean), dim=-1)

        # compute the trace term
        w = torch.einsum(
            "ip, pj, btjq -> btiq", weight.T, weight, torch.diag_embed(x_vars)
        )
        trace = torch.sum(torch.diagonal(w, dim1=-2, dim2=-1), dim=-1)

        return 0.5 * (mahalanobis + trace + n * math.log(2 * math.pi))
    else:
        # compute the log-normalizaer
        log_normalizer = torch.sum(torch.log(2 * torch.pi * y_vars), dim=-1)

        # compute the mahalanobis term
        mahalanobis = torch.sum(torch.square(y - y_mean).div(y_vars), dim=-1)

        # compute the trace term
        _lt = torch.matmul(weight.t(), torch.diag_embed(1 / y_vars))
        _rt = torch.matmul(weight, torch.diag_embed(x_vars))
        trace = torch.sum(torch.diagonal(_lt @ _rt, dim1=-2, dim2=-1), dim=-1)

        return 0.5 * (log_normalizer + mahalanobis + trace)


@torch.jit.script
def recompute_covar(covar: torch.Tensor) -> torch.Tensor:
    """Recompute the covariance matrix for numerical stability.

    Args:
        covar (torch.Tensor): The covariance matrix to recompute.

    Returns:
        torch.Tensor: The recomputed covariance matrix.
    """
    c = 0.5 * torch.add(covar, covar.transpose(-2, -1))
    d, u = torch.linalg.eigh(c)
    d = torch.clamp(d, min=1e-6)
    new_covar = torch.matmul(
        torch.matmul(u, torch.diag_embed(d)), u.transpose(-2, -1)
    )

    return new_covar
