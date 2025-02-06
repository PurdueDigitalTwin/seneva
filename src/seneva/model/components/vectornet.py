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
"""Customized implementation of modules in the `VectorNet` model."""
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.pool import avg_pool_x, max_pool_x

from seneva.model.layers import layer_norm, linear

__all__ = ["VectorNetSubGraphLayer", "VectorNetSubGraph"]


class VectorNetSubGraphLayer(nn.Module):
    """Subgraph layer from the VectorNet model.

    .. note::

        This layer is implemented from "Vectornet: Encoding hd maps and agent
        dynamics from vectorized representation." by Gao et al. The layer is
        a variant of the PointNet layer, with max pooling and concatenation
        for message aggregation.
        .. math::
            \\boldsymbol{v}_i^{(l+1)}\\leftarrow\\text{cat}(\\text{MLP}_l
            (\\boldsymbol{v}_i^{(l)}), \\text{pool}(\\text{MLP}_l(
            \\boldsymbol{v}^{(l)}))`.
    """

    in_dims: int
    """int: Input feature dimensionality."""
    pool: Literal["avg", "max"]
    """Literal["avg", "max"]: Pooling method."""

    def __init__(
        self,
        in_dims: int,
        hidden_size: int,
        pool: Literal["avg", "max"],
    ) -> None:
        """Construct a `VectorSubGraphLayer` instance.

        Args:
            in_dims (int): Input feature dimensionality.
            hidden_size (int): Hidden layer size.
            pool (Literal["avg", "max"]): Pooling method. Either "avg" for
                average-pooling or "max" for max-pooling.
        """
        super().__init__()

        # save arguments
        self.in_dims = in_dims
        self.pool = pool

        # initialize the linear layers
        self._mlp = nn.Sequential(
            linear(in_dims, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(),
            linear(hidden_size, in_dims // 2),
        )

    def forward(
        self, x: torch.Tensor, cluster: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the subgraph layer.

        Args:
            x (torch.Tensor): Input vector features of shape `(N, E)`.
            cluster (torch.Tensor): Vector clusters of shape `(N,)`.
            batch (torch.Tensor): Batch indices of shape `(N,)`.

        Returns:
            torch.Tensor: Output vector features.
        """
        assert x.size(-1) == self.in_dims, "Invalid input dimensions."
        assert (
            x.size(0) == cluster.size(0) == batch.size(0)
        ), "Mismatched input sizes."

        # update the node features
        out = self._mlp.forward(x)

        # aggregate the node features
        if self.pool == "avg":
            aggr, _ = avg_pool_x(cluster=cluster.long(), x=out, batch=batch)
        else:
            aggr, _ = max_pool_x(cluster=cluster.long(), x=out, batch=batch)

        # concatenate the node features
        out = torch.cat([out, aggr[cluster]], dim=-1)

        return out


class VectorNetSubGraph(nn.Module):
    """Subgraph module from the VectorNet model.

    .. note::

        This module is implemented from "VectorNet: Encoding HD Maps and Agent
        Dynamics from Vectorized Representation" by Gao et al. (2020). The
        module consists of a cascade of subgraph layers for feature encoding.
    """

    in_dims: int
    """int: Input vector feature dimensionality."""
    pool: Literal["avg", "max"]
    """Literal["avg", "max"]: Pooling method for the subgraph."""

    _pre_mlp: nn.Sequential
    """nn.Sequential: The MLP for pre-projection of input node features."""
    _layers: nn.ModuleDict
    """nn.ModuleDict: The layers of the subgraph."""
    _norm: nn.LayerNorm
    """nn.LayerNorm: The layer normalization before the output."""

    def __init__(
        self,
        in_dims: int,
        hidden_size: int,
        num_layers: int,
        pool: Literal["avg", "max"] = "max",
    ) -> None:
        """Construct a `VectorSubGraph` module.

        Args:
            in_dims (int): Input vector feature dimensionality.
            hidden_size (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden subgraph layers.
            pool (Literal["avg", "max"]): Pooling method for the subgraph.
                Either "avg" or "max". Defaults to `"max"`.
        """
        super().__init__()

        # save arguments
        self.in_dims = in_dims
        assert pool in ["avg", "max"], (
            "Unsupported pooling method."
            "Expected 'avg' or 'max', "
            f"but got '{pool}'."
        )
        self.pool = pool

        # construct the pre-MLP module
        self._pre_mlp = nn.Sequential(
            linear(in_dims, hidden_size),
            nn.SiLU(),
            linear(hidden_size, hidden_size),
        )

        # construct the cascade of subgraph layers
        self._layers = nn.ModuleDict(
            {
                f"subgraphlayer_{i}": VectorNetSubGraphLayer(
                    in_dims=hidden_size, hidden_size=hidden_size, pool=pool
                )
                for i in range(1, num_layers + 1)
            }
        )

        self._norm = layer_norm(normalized_shape=hidden_size)

    def forward(
        self, x: torch.Tensor, cluster: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the subgraph module.

        Args:
            x (torch.Tensor): Input node features of shape `(N, in_dims)`.
            cluster (torch.Tensor): Cluster indices of shape `(N, )`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output node features and
                their indices.
        """
        assert x.size(-1) == self.in_dims, "Invalid input dimensions."
        assert (
            x.size(0) == cluster.size(0) == batch.size(0)
        ), "Mismatched number of nodes."
        x, cluster = x.float(), cluster.long()

        # forward pass through the pre-MLP
        out = self._pre_mlp.forward(x)

        # forward pass through the subgraph layers
        for layer in self._layers.values():
            assert isinstance(layer, VectorNetSubGraphLayer)
            out = layer.forward(x=out, cluster=cluster, batch=batch)

        # apply pooling to get the polyline features
        if self.pool == "avg":
            out, batch = avg_pool_x(cluster=cluster, x=out, batch=batch)
        else:
            out, batch = max_pool_x(cluster=cluster, x=out, batch=batch)

        out = self._norm.forward(out)

        return out, batch
