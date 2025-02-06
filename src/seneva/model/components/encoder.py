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
"""Graph-based encoder module for the NeVA model."""
import torch
import torch.nn as nn

from seneva.data.base import PolylineData
from seneva.model.components.attention import GlobalGraphLayer
from seneva.model.components.vectornet import VectorNetSubGraph
from seneva.model.functional import sinusoidal_pe


@torch.jit.script
def get_current_state(
    state: torch.Tensor, cluster: torch.Tensor, timestamps: torch.Tensor
) -> torch.Tensor:
    unique_cluster = torch.unique(cluster)
    output = torch.zeros(
        (len(unique_cluster), state.size(-1)),
        dtype=state.dtype,
        device=state.device,
    )
    for i, idx in enumerate(unique_cluster):
        max_idx = timestamps[cluster == idx].argmax()
        output[i] = state[cluster == idx][max_idx]

    return output


class Encoder(nn.Module):
    """Encoder module wrapping MapNet and MotionNet for feature encoding."""

    out_channels: int
    """int: Output feature dimensionality of the encoder."""
    num_global_layers: int
    """int: Number of global interaction layers."""

    _mapnet: VectorNetSubGraph
    """VectorNetSubGraph: VectorNet subgraph for map feature encoding."""
    _motionnet: VectorNetSubGraph
    """VectorNetSubGraph: VectorNet Subgraph for motion feature encoding."""
    _global_layers: nn.ModuleList
    """nn.ModuleList: global interaction layers."""

    def __init__(
        self,
        map_in_dims: int,
        map_hidden_size: int,
        map_num_layers: int,
        motion_in_dims: int,
        motion_hidden_size: int,
        motion_num_layers: int,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_global_layers: int = 3,
        *args,
        **kwargs,
    ) -> None:
        """Construct a new `Encoder` module.

        Args:
            map_in_dims (int): Input map feature dimensionality.
            map_hidden_size (int): Map subgraph hidden layer dimensions.
            map_num_layers (int): Number of hidden layers in map subgraph.
            motion_in_dims (int): Input motion feature dimensionality.
            motion_hidden_size (int): Track subgraph hidden layer dimensions.
            motion_num_layers (int): Number of hidden layers in track subgraph.
            dropout (float): Attention dropout rate. Defaults to :math:`0.1`.
            num_heads (int): Number of attention heads. Defaults to :math:`8`.
        """
        super().__init__()

        # save arguments
        self.out_feature = motion_hidden_size

        # initialize map encoder
        self._mapnet = VectorNetSubGraph(
            in_dims=map_in_dims,
            hidden_size=map_hidden_size,
            num_layers=map_num_layers,
        )

        # initialize motion encoder
        self._motionnet = VectorNetSubGraph(
            in_dims=motion_in_dims,
            hidden_size=motion_hidden_size,
            num_layers=motion_num_layers,
        )

        # initialize agent-to-map attention layer
        self._global_layers = nn.ModuleList(
            [
                GlobalGraphLayer(
                    agent_feature_dim=motion_hidden_size,
                    map_feature_dim=map_hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_global_layers)
            ]
        )

    def forward(self, data: PolylineData) -> torch.Tensor:
        # encode map features
        map_batch: torch.LongTensor = data.get(
            "x_map_batch", torch.zeros_like(data.x_map[:, 0].long())
        )
        map_feats, map_batch = self._mapnet.forward(
            x=data.x_map, cluster=data.map_cluster, batch=map_batch
        )
        assert len(map_batch) == len(map_feats), "Mismatched sizes."

        # encode motion features
        motion_batch: torch.LongTensor = data.get(
            "x_motion_batch", torch.zeros_like(data.x_motion[:, 0].long())
        )
        track_feats, track_batch = self._motionnet.forward(
            x=data.x_motion, cluster=data.motion_cluster, batch=motion_batch
        )
        assert len(track_batch) == len(track_feats), "Mismatched sizes."

        # only keep the motion features of the tracks to predict
        with torch.no_grad():
            tar_filter = torch.isin(data.track_ids, data.tracks_to_predict)
            tar_filter = tar_filter.view(-1)

        # compute positional embeddings
        map_pe = sinusoidal_pe(pos=data.map_pos, dim=map_feats.size(-1))
        map_feats = map_feats + map_pe
        track_pe = sinusoidal_pe(pos=data.track_pos, dim=track_feats.size(-1))
        track_feats = track_feats + track_pe

        for layer in self._global_layers:
            assert isinstance(layer, GlobalGraphLayer)
            track_feats, map_feats = layer.forward(
                agent_x=track_feats,
                map_x=map_feats,
                agent_batch=track_batch,
                map_batch=map_batch,
            )

        return track_feats[tar_filter]

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"lanenet={self._mapnet}",
                f"motionnet={self._motionnet}",
            ]
        )
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)
