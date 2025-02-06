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
"""Performance metrics for the INTERACTION data."""
from typing import Optional, Union

import torch
from numpy.typing import NDArray
from torch import FloatTensor, Tensor
from torchmetrics import Metric

from seneva.data.transform import project_poses


class MinAverageDisplacementError(Metric):
    """Compute minimum mean displacement error for multi-modal predictions."""

    total: FloatTensor
    """FloatTensor: Total min. mean displacement error of all the samples."""
    count: FloatTensor
    """FloatTensor: Total number of samples."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: Union[NDArray, Tensor],
        target: Union[NDArray, Tensor],
        *args,
        **kwargs,
    ) -> None:
        """Compute and update current minimum mean displacement error tracker.

        Args:
            preds (Union[NDArray, Tensor]): Prediction of shape `[N, M, T, F]`.
            target (Union[NDArray, Tensor]): Ground truth of shape `[N, T, F]`.
        """
        if not isinstance(preds, Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, Tensor):
            target = torch.from_numpy(target)
        preds, target = preds.float(), target.float()

        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) >= 2
        ), "Invalid target trajectory!"

        with torch.no_grad():
            preds = preds[..., 0:2].float().to(self.device)
            if preds.ndim == 3:
                # unsqueeze modal dimensionality if necessary
                preds.unsqueeze(1)
            target = target[..., 0:2].float().unsqueeze(1).to(self.device)

            displacement_errors: torch.Tensor = torch.norm(
                preds - target, p=2, dim=-1
            )  # NOTE: shape = [N, M, T]
            min_error, _ = displacement_errors.mean(-1).min(1)
            self.total += min_error.sum()
            self.count += target.size(0)

    def compute(self) -> FloatTensor:
        return self.total.float() / self.count


class MinFinalDisplacementError(Metric):
    """Compute minimum final displacement error for multi-modal predictions."""

    total: FloatTensor
    """FloatTensor: Total min. final displacement error of all the samples."""
    count: FloatTensor
    """FloatTensor: Total number of samples."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: Union[NDArray, Tensor],
        target: Union[NDArray, Tensor],
        *args,
        **kwargs,
    ) -> None:
        """Compute and update current minimum final displacement error tracker.

        Args:
            preds (Union[NDArray, Tensor]): Prediction of shape `[N, M, T, F]`.
            target (Union[NDArray, Tensor]): Ground truth of shape `[N, T, F]`.
        """
        if not isinstance(preds, Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, Tensor):
            target = torch.from_numpy(target)
        preds, target = preds.float(), target.float()

        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) >= 2
        ), "Invalid target trajectory!"

        with torch.no_grad():
            preds = preds[..., 0:2].float().to(self.device)
            if preds.ndim == 3:
                # unsqueeze modal dimensionality if necessary
                preds.unsqueeze(1)
            target = target[..., 0:2].float().unsqueeze(1).to(self.device)

            displacement_errors: torch.Tensor = torch.norm(
                preds - target, p=2, dim=-1
            )  # NOTE: shape = [N, M, T]
            min_error, _ = displacement_errors[..., -1].min(1)
            self.total += min_error.sum()
            self.count += target.size(0)

    def compute(self) -> FloatTensor:
        return self.total.float() / self.count


class MissRate(Metric):
    """Compute miss rate for multi-modal predictions."""

    total: FloatTensor
    """FloatTensor: Total number of miss case of all the samples."""
    count: FloatTensor
    """FloatTensor: Total number of samples."""
    use_piecewise_threshold: bool
    """bool: whether to use piecewise longitudinal thresholds for miss rate."""

    def __init__(self, use_piecewise_threshold: bool = True) -> None:
        super().__init__()
        self.use_piecewise_threshold = use_piecewise_threshold
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        target: Tensor,
        anchor: Tensor,
        batch: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> None:
        if not isinstance(preds, Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(target, Tensor):
            target = torch.from_numpy(target)
        preds, target = preds.float(), target.float()

        assert (
            preds.ndim >= 3 and preds.size(-1) >= 2
        ), "Invalid prediction input!"
        assert (
            target.ndim == 3 and target.size(-1) == 3
        ), "Invalid target trajectory!"
        assert (
            anchor.ndim == 2 and anchor.size(-1) == 3
        ), "Invalid coordinate anchors!"

        if batch is None:
            batch = torch.zeros(
                target.size(0), device=self.device, dtype=torch.long
            )

        # validate the correctness of this implementation.
        if self.use_piecewise_threshold:
            with torch.no_grad():
                gt_vel = torch.norm(target[..., -1:, 3:5], dim=-1).clone()
                gt_hdg = target[..., -1:, 2].clone()
                cosine, sine = gt_hdg.cos(), gt_hdg.sin()
                rot_mat = torch.cat(
                    [
                        torch.cat([cosine, -sine], dim=-1).unsqueeze(1),
                        torch.cat([sine, cosine], dim=-1).unsqueeze(1),
                    ],
                    dim=1,
                ).to(self.device)

                preds = (
                    preds[..., -1, 0:3].clone().float().to(self.device).clone()
                )
                if preds.ndim == 3:
                    preds.unsqueeze(1)
                target = (
                    target[..., -1, 0:3]
                    .clone()
                    .float()
                    .unsqueeze(1)
                    .to(self.device)
                )

                # project back to global frame
                for idx in torch.unique(batch):
                    _filter = batch == idx.item()
                    preds[_filter] = (
                        project_poses(
                            preds[_filter, ...].view(-1, 3),
                            anchor[idx.item()].to(self.device),
                            precision=torch.float32,
                            inverse=True,
                        )
                        .view(_filter.sum().item(), -1, 3)
                        .clone()
                        .squeeze(1)
                    )
                    target[_filter] = (
                        project_poses(
                            target[_filter].view(-1, 3),
                            anchor[idx.item()].to(self.device),
                            precision=torch.float32,
                            inverse=True,
                        )
                        .view(_filter.sum().item(), -1, 3)
                        .clone()
                    )

                if preds.ndim == 2:
                    preds = preds.unsqueeze(1)
                if target.ndim == 2:
                    target = target.unsqueeze(1)
                displ = torch.bmm(preds[..., 0:2] - target[..., 0:2], rot_mat)

                lng_thld = torch.zeros_like(gt_vel)
                lng_thld = torch.where(gt_vel < 1.4, 1.0, lng_thld)
                lng_thld = torch.where(
                    (gt_vel >= 1.4) & (gt_vel < 11.0),
                    1.0 + torch.div(gt_vel - 1.4, 11.0 - 1.4),
                    lng_thld,
                )
                lng_thld = torch.where(gt_vel > 11.0, 2.0, lng_thld)

                miss_flags = torch.add(
                    displ[..., 0].abs() > lng_thld, displ[..., 1].abs() > 1.0
                )
        else:
            with torch.no_grad():
                preds = preds[..., 0:2].float().to(self.device)
                if preds.ndim == 3:
                    # unsqueeze modal dimensionality if necessary
                    preds.unsqueeze(1)
                target = target[..., 0:2].float().unsqueeze(1).to(self.device)

                displacement_errors: torch.Tensor = torch.norm(
                    preds - target, p=2, dim=-1
                )  # NOTE: shape = [N, M, T]
                fde = displacement_errors[..., -1]  # NOTE: shape = [N, M]
                miss_flags = fde > 2.0
        self.total += torch.sum(torch.all(miss_flags, dim=1))
        self.count += target.size(0)

    def compute(self) -> FloatTensor:
        return self.total.float() / self.count * 100
