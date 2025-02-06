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
"""Lightning DataModule wrappers."""
from typing import Optional

from seneva.data.base import BaseDataModule
from seneva.data.dataset import Argoverse2Dataset, INTERACTIONDataset
from seneva.data.subsampler import INTERACTIONSubsampler
from seneva.data.typing import Transforms


class Argoverse2DataModule(BaseDataModule):
    """A wrapper of `LightningDataModule` for Argoverse 2 dataset."""

    def __init__(
        self,
        root: str,
        radius: float = None,
        transform: Optional[Transforms] = None,
        enable_train: bool = True,
        enable_val: bool = True,
        enable_test: bool = False,
        batch_size: int = 64,
        num_workers: int = None,
        pin_memory: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor function.

        Args:
            root (str): root directory of the INTERACTION dataset.
            radius (Optional[float], optional): query range in meters.
                Defaults to `None`.
            transform (Optional[Transforms], optional): data transform modules.
                Defaults to `None`.
            force_data_cache (bool, optional): if save cache tensors to local.
                Defaults to `False`.
            batch_size (int, optional): batch size. Defaults to 64.
            num_workers (int, optional): number of workers. Defaults to 0.
            pin_memory (bool, optional): if pin memory. Defaults to False.
        """
        if enable_train:
            train_dataset = Argoverse2Dataset(
                root=root,
                split="train",
                radius=radius,
                transform=transform,
            )
        else:
            train_dataset = None

        if enable_val:
            val_dataset = Argoverse2Dataset(
                root=root,
                split="val",
                radius=radius,
                transform=transform,
            )
        else:
            val_dataset = None

        if enable_test:
            test_dataset = Argoverse2Dataset(
                root=root,
                split="test",
                radius=radius,
                transform=transform,
            )
        else:
            test_dataset = None

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            test_dataset=test_dataset,
            *args,
            **kwargs,
        )


class INTERACTIONDataModule(BaseDataModule):
    """An implementation of `BaseDataModule` for INTERACTION dataset."""

    def __init__(
        self,
        root: str,
        challenge_type: str,
        subsampler: Optional[INTERACTIONSubsampler] = None,
        radius: Optional[float] = None,
        transform: Optional[Transforms] = None,
        train_on_full_data: bool = False,
        enable_train: bool = True,
        enable_val: bool = True,
        enable_test: bool = False,
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor function.

        Args:
            root (str): root directory of the INTERACTION dataset.
            challenge_type (str): name of the challenge, either `single-agent`,
                `conditional-single-agent`, `multi-agent`, or
                `conditional-multi-agent`.
            subsampler (Optional[INTERACTIONSubsampler], optional): subsampler
                for the dataset. Defaults to `None`.
            radius (Optional[float], optional): query range in meters.
                Defaults to `None`.
            transform (Optional[Transforms], optional): data transform modules.
                Defaults to `None`.
            train_on_full_data (bool, optional): if train on full data.
                Defaults to `False`.
            enable_train (bool, optional): if enable train. Defaults to `True`.
            enable_val (bool, optional): if enable val. Defaults to `True`.
            enable_test (bool, optional): if enable test. Defaults to `False`.
            force_data_cache (bool, optional): if save cache tensors to local.
                Defaults to `False`.
            batch_size (int, optional): batch size. Defaults to 64.
            num_workers (int, optional): number of workers. Defaults to 0.
            pin_memory (bool, optional): if pin memory. Defaults to False.
        """
        if enable_train:
            train_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="trainval" if train_on_full_data else "train",
                radius=radius,
                subsampler=subsampler or INTERACTIONSubsampler(),
                transform=transform,
                num_workers=num_workers,
            )
        else:
            train_dataset = None

        if enable_val:
            val_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="val",
                radius=radius,
                subsampler=subsampler or INTERACTIONSubsampler(),
                transform=transform,
                num_workers=num_workers,
            )
        else:
            val_dataset = None

        if enable_test:
            test_dataset = INTERACTIONDataset(
                root=root,
                challenge_type=challenge_type,
                split="test",
                radius=radius,
                subsampler=subsampler or INTERACTIONSubsampler(),
                transform=transform,
                num_workers=num_workers,
            )
        else:
            test_dataset = None

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            test_dataset=test_dataset,
            *args,
            **kwargs,
        )
