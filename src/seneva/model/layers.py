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
"""Common layers with customizable initializations."""
from typing import Any, Literal, Tuple

import torch
import torch.nn as nn


def variance_scaling(
    tensor: torch.Tensor,
    scale: float = 1.0,
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
    distribution: Literal["uniform", "normal", "truncated_normal"] = "normal",
) -> None:
    """Initialize the tensor in-place with variance scaling.

    This function implements the variance scaling initialization method
    as in the TensorFlow library :url:`https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling` as well as in the JAX library :url:`https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html`.

    Args:
        tensor (torch.Tensor): The input tensor to be initialized in-place.
        scale (float, optional): The scaling factor (positive float).
            Defaults to :math:`1.0`.
        mode (Literal["fan_in", "fan_out", "fan_avg"], optional): One of the
            `"fan_in"`, `"fan_out"`, or `"fan_avg"`. Defaults to `"fan_in"`.
        distribution (Literal["uniform", "normal", "truncated_normal"],
            optional): One of `"uniform"`, `"normal"`, or
            `"truncated_normal"`. Defaults to "normal".
    """
    assert (
        isinstance(scale, float) and scale >= 0.0
    ), "The scale factor must be non-negative."
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor=tensor)
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_out":
        n = fan_out
    elif mode == "fan_avg":
        n = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")
    std = (max(1e-10, scale) / n) ** 0.5
    if distribution == "uniform":
        nn.init.uniform_(tensor, a=-std, b=std)
    elif distribution == "normal":
        nn.init.normal_(tensor, mean=0.0, std=std)
    elif distribution == "truncated_normal":
        a, b = -2.0 * std, 2.0 * std
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def linear(
    in_features: int,
    out_features: int,
    init_scale: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> nn.Linear:
    """Create a linear layer with custom initialization.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        init_scale (float): The scale factor for the initialization.
        *args, **kwargs: Additional arguments for the linear layer.

    Returns:
        nn.Linear: The initialized linear layer.
    """
    layer = nn.Linear(in_features=in_features, out_features=out_features)
    if hasattr(layer, "weight"):
        variance_scaling(layer.weight, scale=init_scale)
    if hasattr(layer, "bias"):
        nn.init.constant_(layer.bias, val=0.0)

    return layer


def layer_norm(
    normalized_shape: Tuple[int, ...],
    eps: float = 1e-9,
    *args: Any,
    **kwargs: Any,
) -> nn.LayerNorm:
    """Create a layer normalization layer.

    Args:
        normalized_shape (Tuple[int, ...]): The shape of the input tensor.
        eps (float, optional): The epsilon value for numerical stability.
            Defaults to :math:`1e-9`.
        *args, **kwargs: Additional arguments for the normalization layer.

    Returns:
        nn.LayerNorm: The layer normalization layer.
    """
    layer = nn.LayerNorm(
        normalized_shape=normalized_shape, eps=eps, *args, **kwargs
    )
    if hasattr(layer, "weight"):
        nn.init.ones_(layer.weight)
    if hasattr(layer, "bias"):
        nn.init.zeros_(layer.bias)

    return layer
