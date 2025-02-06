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
"""Helper functions for interaction data processing."""
from typing import Optional

import torch
from torch_geometric.transforms import BaseTransform

from .base import PolylineData


@torch.jit.script
def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    """Wraps angle to the range :math:`[-\\pi, \\pi]`.

    Args:
        angle (torch.Tensor): Input angle in radians.

    Returns:
        torch.Tensor: Wrapped angle in the range :math:`[-\\pi, \\pi]`.
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


@torch.jit.script
def project_poses(
    poses: torch.Tensor,
    anchor: torch.Tensor,
    precision: Optional[torch.dtype] = None,
    inverse: bool = False,
) -> torch.Tensor:
    """Affine transform input `poses` to new system centered about `anchor`.

    Args:
        poses (Tensor): local states, of shape `[*, 3]` or `[3, ]`.
        anchor (Tensor): global anchor, of shape `[*, 3]` or `[3, ]`.
        precision (Optional[torch.dtype], optional): the precision of the
            computation. If `None`, use `poses` precision. Defaults to `None`.
        inverse (bool, optional): if apply inverse transform. If `True`,
            project the target-centric state back to the global frame.
            Defaults to `False`.

    Returns:
        Tensor: global states, of shape `[*, 3]`.
    """
    assert (
        poses.size(-1) == 3
    ), "Invalid input `poses`. Expect last dimension to be 3."
    assert (
        anchor.size(-1) == 3
    ), "Invalid input `anchor`. Expect last dimension to be 3."
    if poses.ndim == 2 and anchor.ndim == 1:
        # expand anchor to match poses
        anchor = anchor.view(1, 3)
    assert poses.ndim == anchor.ndim, "Unmatched input shapes."

    if precision is None:
        precision = poses.dtype
    output = torch.zeros_like(poses, dtype=precision)

    # unpack local states and global anchor
    x, y, theta = poses.unbind(dim=-1)
    x0, y0, theta0 = anchor.unbind(dim=-1)

    # compute projection
    cosine = torch.cos(theta0)
    sine = torch.sin(theta0)

    if inverse:
        x_new = cosine * x - sine * y + x0
        y_new = sine * x + cosine * y + y0
        theta_new = wrap_angle_torch(theta + theta0)
    else:
        x_new = cosine * (x - x0) + sine * (y - y0)
        y_new = -sine * (x - x0) + cosine * (y - y0)
        theta_new = wrap_angle_torch(theta - theta0)
    output = torch.stack((x_new, y_new, theta_new), dim=-1, out=output)

    return output


@torch.jit.script
def project_velocities(
    vels: torch.Tensor,
    anchor: torch.Tensor,
    precision: Optional[torch.dtype] = None,
    inverse: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Rotate input `vels` to new system centered about `anchor`.

    Args:
        vels (Tensor): local velocities, of shape `[*, 2]` or `[2, ]`.
        anchor (Tensor): global anchor, of shape `[*, 3]` or `[3, ]`.
        precision (Optional[torch.dtype], optional): the precision of the
            computation. If `None`, use `poses` precision. Defaults to `None`.
        inverse (bool, optional): if apply inverse transform. If `True`,
            project the target-centric state back to the global frame.
            Defaults to `False`.
        normalize (bool, optional): if normalize the raw velocities.
            Defaults to `True`.

    Returns:
        Tensor: global velocities, of shape `[*, 2]`.
    """
    assert (
        vels.size(-1) == 2
    ), "Invalid input `vels`. Expect last dimension to be 2."
    assert (
        anchor.size(-1) == 3
    ), "Invalid input `anchor`. Expect last dimension to be 3."
    if vels.ndim == 2 and anchor.ndim == 1:
        # expand anchor to match velocities
        anchor = anchor.view(1, 3)
    assert vels.ndim == anchor.ndim, "Unmatched input shapes."

    if precision is None:
        precision = vels.dtype

    # unpack local velocities and global anchor
    vx, vy = vels.unbind(dim=-1)
    _, _, theta0 = anchor.unbind(dim=-1)

    # compute projection
    cosine = torch.cos(theta0)
    sine = torch.sin(theta0)

    if inverse:
        if normalize:
            vx, vy = vx * 40.0, vy * 40.0
        vx_new = cosine * vx - sine * vy
        vy_new = sine * vx + cosine * vy
    else:
        if normalize:
            vx, vy = vx / 40.0, vy / 40.0
        vx_new = cosine * vx + sine * vy
        vy_new = -sine * vx + cosine * vy
    output = torch.stack((vx_new, vy_new), dim=-1).to(precision)

    return output


class TargetCentricTransform(BaseTransform):
    """Project the global state to the target-centric frame."""

    def __call__(
        self, data: PolylineData, inverse: bool = False
    ) -> PolylineData:
        """Project the global state to the target-centric frame.

        Args:
            data (PolylineData): data to be transformed.
            inverse (bool, optional): if apply inverse transform. If `True`,
                project the target-centric state back to the global frame.
                Defaults to `False`.

        Returns:
            PolylineData: Transformed data.
        """
        assert isinstance(
            data, PolylineData
        ), f"Unsupported data type: {type(data)}"

        with torch.no_grad():
            anchor = data.anchor
            data.x_map[:, 0:2] = project_poses(
                poses=torch.hstack(
                    [
                        data.x_map[:, 0:2],
                        torch.zeros_like(data.x_map[:, 0:1]),
                    ]
                ),
                anchor=anchor,
                precision=torch.float32,
                inverse=inverse,
            )[:, 0:2]
            data.x_motion[:, 0:3] = project_poses(
                data.x_motion[:, 0:3],
                anchor,
                precision=torch.float32,
                inverse=inverse,
            )
            data.x_motion[:, 3:5] = project_velocities(
                data.x_motion[:, 3:5],
                anchor,
                precision=torch.float32,
                inverse=inverse,
            )

            if len(data.y_motion) > 0:
                if data.y_motion.ndim == 3:
                    b, h, f = data.y_motion.size()
                    data.y_motion = data.y_motion.view(-1, f)
                    data.y_motion[:, 0:3] = project_poses(
                        data.y_motion[:, 0:3],
                        anchor,
                        precision=torch.float32,
                        inverse=inverse,
                    )
                    data.y_motion[:, 3:5] = project_velocities(
                        data.y_motion[:, 3:5],
                        anchor,
                        precision=torch.float32,
                        inverse=inverse,
                    )
                    data.y_motion = data.y_motion.view(b, h, f)
                else:
                    data.y_motion[:, 0:3] = project_poses(
                        data.y_motion[:, 0:3],
                        anchor,
                        precision=torch.float32,
                        inverse=inverse,
                    )
                    data.y_motion[:, 3:5] = project_velocities(
                        data.y_motion[:, 3:5],
                        anchor,
                        precision=torch.float32,
                        inverse=inverse,
                    )

        return data

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


class TargetReshapeTransform(BaseTransform):
    """Reshape target motion states."""

    horizon: int
    """int: The prediction horizon in number of waypoints."""

    def __init__(self, horizon: int = 30) -> None:
        super().__init__()
        self.horizon = horizon

    def __call__(self, data: PolylineData) -> PolylineData:
        assert isinstance(
            data, PolylineData
        ), f"Unsupported type: {type(data)}."
        agent_ids: torch.LongTensor = data.y_cluster.unique()

        size = (
            agent_ids.size(0),
            self.horizon,
            data.y_motion.size(-1),
        )
        new_y_motion = torch.zeros(
            size=size, device=data.y_motion.device, dtype=torch.float32
        )
        new_y_cluster = torch.zeros(
            size=size[0:1], device=data.y_cluster.device, dtype=torch.long
        )
        new_y_valid = torch.zeros(
            size=size[0:2], device=data.y_valid.device, dtype=torch.bool
        )

        for idx, cluster in enumerate(agent_ids):
            mask = data.y_cluster == cluster
            num_states = torch.nonzero(mask).size(0)
            new_y_motion[idx, 0:num_states] = data.y_motion[mask]
            new_y_cluster[idx] = cluster
            new_y_valid[idx, 0:num_states] = True

        data.y_motion = new_y_motion
        data.y_cluster = new_y_cluster
        data.y_valid = new_y_valid

        return data

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)
