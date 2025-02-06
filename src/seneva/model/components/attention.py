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
"""Attention modules for the neural network model."""
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from seneva.model.layers import layer_norm, linear, variance_scaling

__all__ = ["GlobalGraphLayer"]


class AttentionLayer(nn.Module):
    """Multi-head attention block with feed-forward network."""

    # ----------- public attributes ----------- #
    query_dims: int
    """int: The query feature dimensionality."""
    key_dims: int
    """int: The key feature dimensionality."""
    value_dims: int
    """int: The value feature dimensionality."""

    # ----------- private attributes ----------- #
    _mha_layer: nn.MultiheadAttention
    """nn.MultiheadAttention: The multi-head attention layer."""
    _mha_norm: nn.LayerNorm
    """nn.LayerNorm: The layer normalization layer."""
    _ffn: nn.Sequential
    """nn.Sequential: The feed-forward network layer."""
    _ffn_norm: nn.LayerNorm
    """nn.LayerNorm: The layer normalization layer."""

    def __init__(
        self,
        query_dims: int,
        key_value_dims: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Construct an `MHAGlobalLayer` instance.

        Args:
            query_dims (int): The query feature dimensionality.
            key_value_dims (Optional[int], optional): The dimensionality of
                key and value features. Defaults to `None` for self-attention.
            num_heads (int, optional): The number of heads in the multi-head
                attention operator. Defaults to :math:`8`.
            dropout (float, optional): Dropout rate. Defaults to :math:`0.1`.
            bias (bool, optional): Whether to use bias in the linear layers.
                Defaults to `True`.
        """
        super().__init__()

        # save arguments
        self.query_dims = query_dims
        self.key_value_dims = key_value_dims

        # construct the multi-head attention layer
        self._query_norm = layer_norm(normalized_shape=query_dims)
        if self.key_value_dims is None:
            self.key_value_dims = query_dims
            self._key_val_norm = self._query_norm
        else:
            self._key_val_norm = layer_norm(normalized_shape=key_value_dims)
        self._mha_layer = nn.MultiheadAttention(
            embed_dim=query_dims,
            kdim=key_value_dims,
            vdim=key_value_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )

        # construct the fully connected layers
        self._ffn = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", linear(query_dims, query_dims * 4)),
                    ("relu", nn.SiLU()),
                    ("linear_2", linear(query_dims * 4, query_dims)),
                ]
            )
        )
        self._ffn_norm = layer_norm(normalized_shape=query_dims)

        # initialize the parameters
        self.reset_parameters()

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the global layer.

        Args:
            query (torch.Tensor): Query feature tensor.
            key_value (Optional[torch.Tensor], optional): Key and value
                feature tensor. Defaults to `None` for self-attention.
            *args, **kwargs: Additional arguments and keyword arguments for
                the multi-head attention layer.

        Returns:
            torch.Tensor: Output tensor.
        """
        assert query.size(-1) == self.query_dims, "Invalid query dims."
        if key_value is None:
            key = query
            value = query
        else:
            key = key_value
            value = key_value
        assert key.size(-1) == self.key_value_dims, "Invalid key dims."
        query, key, value = query.float(), key.float(), value.float()

        # forward pass multi-head attention
        mha_out, _ = self._mha_layer.forward(
            query=self._query_norm.forward(query),
            key=self._key_val_norm.forward(key),
            value=self._key_val_norm.forward(value),
            need_weights=False,
            *args,
            **kwargs,
        )
        mha_out = query + mha_out

        # forward pass feed-forward network
        ffn_out = self._ffn.forward(self._ffn_norm.forward(mha_out))
        ffn_out = mha_out + ffn_out

        return ffn_out

    def reset_parameters(self) -> None:
        # initialize the multi-head attention layer
        if self._mha_layer.in_proj_weight is not None:
            variance_scaling(self._mha_layer.in_proj_weight)
        else:
            variance_scaling(self._mha_layer.q_proj_weight)
            variance_scaling(self._mha_layer.k_proj_weight)
            variance_scaling(self._mha_layer.v_proj_weight)
        if self._mha_layer.in_proj_bias is not None:
            nn.init.zeros_(self._mha_layer.in_proj_bias)
        variance_scaling(self._mha_layer.out_proj.weight)
        if self._mha_layer.out_proj.bias is not None:
            nn.init.zeros_(self._mha_layer.out_proj.bias)
        if self._mha_layer.bias_k is not None:
            nn.init.zeros_(self._mha_layer.bias_k)
        if self._mha_layer.bias_v is not None:
            nn.init.zeros_(self._mha_layer.bias_v)


class GlobalGraphLayer(nn.Module):
    """Encapsulate a cascade of global layer for interaction encoding."""

    _a2m_layer: AttentionLayer
    """AttentionLayer: Global layer to encode agent-to-map interaction."""
    _m2m_layer: AttentionLayer
    """AttentionLayer: Global layer to encode map-to-map interaction."""
    _m2a_layer: AttentionLayer
    """AttentionLayer: Global layer to encode map-to-agent interaction."""
    _a2a_layer: AttentionLayer
    """AttentionLayer: Global layer to encode agent-to-agent interaction."""

    def __init__(
        self,
        agent_feature_dim: int,
        map_feature_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # initialize global attention modules
        self._a2m_layer = AttentionLayer(
            query_dims=map_feature_dim,
            key_value_dims=agent_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self._m2m_layer = AttentionLayer(
            query_dims=map_feature_dim, num_heads=num_heads, dropout=dropout
        )
        self._m2a_layer = AttentionLayer(
            query_dims=agent_feature_dim,
            key_value_dims=map_feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self._a2a_layer = AttentionLayer(
            query_dims=agent_feature_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(
        self,
        agent_x: torch.Tensor,
        map_x: torch.Tensor,
        agent_batch: Optional[torch.Tensor] = None,
        map_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_x, map_x = agent_x.float(), map_x.float()
        if agent_batch is None:
            agent_batch = torch.zeros_like(agent_x[:, 0]).long()
        if map_batch is None:
            map_batch = torch.zeros_like(map_x[:, 0]).long()

        # agent-to-lane
        attn_mask = map_batch.unsqueeze(1) != agent_batch.unsqueeze(0)
        map_x = self._a2m_layer.forward(
            query=map_x, key_value=agent_x, attn_mask=attn_mask
        )

        # lane-to-lane
        attn_mask = map_batch.unsqueeze(1) != map_batch.unsqueeze(0)
        map_x = self._m2m_layer.forward(query=map_x, attn_mask=attn_mask)

        # lane-to-agent
        attn_mask = agent_batch.unsqueeze(1) != map_batch.unsqueeze(0)
        agent_x = self._m2a_layer.forward(
            query=agent_x, key_value=map_x, attn_mask=attn_mask
        )

        # agent-to-agent
        attn_mask = agent_batch.unsqueeze(1) != agent_batch.unsqueeze(0)
        agent_x = self._a2a_layer.forward(query=agent_x, attn_mask=attn_mask)

        return agent_x, map_x
