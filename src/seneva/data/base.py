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
"""Definitions of the base classes for the data module."""
import abc
from functools import cached_property
from multiprocessing import cpu_count
from typing import Any, Iterable, Optional, Type

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torch import BoolTensor, FloatTensor, LongTensor
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


# ---------- Data classes ---------- #
class BaseData(abc.ABC, Data):
    """Base implementation for data classes."""

    follow_batch: Iterable[str]
    """Iterable[str]: An iterable of attribute names to follow batch."""


class PolylineData(BaseData):
    """Data class for representing a traffic environment by polylines.

    .. note::

        A polyline data object represents a traffic environment by a collection
        of polylines. Each polyline can either be a map object or an agent's
        trajectory, consisting of a sequence of vectors.
    """

    # map subgraph
    x_map: FloatTensor
    """FloatTensor: The polyline vectors associated with map objects."""
    map_pos: FloatTensor
    """FloatTensor: The representative position of each map object."""
    map_cluster: LongTensor
    """LongTensor: The polyline label of map vectors."""

    # agent observation subgraph
    x_motion: FloatTensor
    """FloatTensor: The trajectory vectors associated with the agents."""
    motion_cluster: LongTensor
    """LongTensor: The agent label of trajectory vectors."""
    motion_timestamps: FloatTensor
    """FloatTensor: The endpoint timestamps of trajectory vectors."""
    track_pos: FloatTensor
    """FloatTensor: The representative position of each agent."""
    track_ids: LongTensor
    """LongTensor: The ground-truth agent id of each trajectory."""

    # target subgraph
    y_motion: FloatTensor
    """FloatTensor: The ground-truth future motion states of the agents."""
    y_valid: BoolTensor
    """BoolTensor: The ground-truth future motion state validity."""
    y_cluster: LongTensor
    """LongTensor: The ground-truth future state cluster as agent indexes."""

    # system info
    anchor: FloatTensor
    """FloatTensor: The anchor as a 2D pose in this sample of shape `[3,]`."""
    tracks_to_predict: LongTensor
    """LongTensor: The agent ids of the tracks to predict."""
    num_agents: LongTensor
    """LongTensor: The number of agents in this sample."""

    # metadata
    follow_batch: Iterable[str] = (
        "x_map",
        "x_motion",
        "track_ids",
        "y_motion",
        "tracks_to_predict",
    )
    """Iterable[str]: An iterable of attribute names to follow batch."""
    sample_idx: LongTensor
    """The sample index of this sample in the source dataset."""

    def is_valid(self) -> bool:
        """Sanity check for the data model.

        Returns:
            bool: True if the data model is valid, False otherwise.
        """
        map_valid = (
            # map data sanity checks
            self.x_map.size(0) > 0
            and self.x_map.size(1) == self.map_feature_dims
            and self.map_cluster.size(0) == self.x_map.size(0)
            and self.map_cluster.min().item() >= 0
        )
        track_valid = (
            # motion state data sanity checks
            self.x_motion.size(0) > 0
            and self.x_motion.size(1) == self.motion_feature_dims
            and self.motion_cluster.size(0) == self.x_motion.size(0)
            and self.motion_cluster.min().item() >= 0
            and self.motion_cluster.max().item() == self.track_ids.size(0) - 1
            and self.motion_timestamps.size(0) == self.x_motion.size(0)
            # track data sanity checks
            and self.track_ids.size(0) == self.motion_cluster.unique().size(0)
            and self.track_ids.min().item() >= 0
            # target data sanity checks
            and (
                torch.all(torch.isin(self.tracks_to_predict, self.y_cluster))
                if len(self.y_cluster) > 0
                else True
            )
        )

        return map_valid and track_valid

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "map_cluster":
            return self.num_map_objects
        elif key == "motion_cluster":
            return self.track_ids.size(0)
        elif key in ["y_cluster", "track_ids", "tracks_to_predict"]:
            return self.num_agents[0].item()
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in ["anchor", "num_agents"]:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    @cached_property
    def num_map_objects(self) -> int:
        """int: The number of map objects in this sample."""
        return len(self.map_cluster.unique())

    @property
    @abc.abstractmethod
    def map_feature_dims(self) -> int:
        """int: The number of features in the map vectors."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def motion_feature_dims(self) -> int:
        """int: The number of features in the motion vectors."""
        raise NotImplementedError


# ---------- Dataset classes ---------- #
class BaseDataset(Dataset):
    """Base implementation for dataset classes."""

    @property
    @abc.abstractmethod
    def data_type(self) -> Type[BaseData]:
        """Type[BaseData]: The type of data class to use."""
        raise NotImplementedError


class BaseDataModule(LightningDataModule):
    """A wrapper of `LightningDataModule` for the base dataset."""

    # ----------- public attributes ----------- #
    data_type: BaseData
    """BaseData: The data type for the dataset."""
    batch_size: int
    """int: The batch size for data loaders."""
    num_workers: int
    """int: The number of workers for data loaders."""
    pin_memory: bool
    """bool: If pin memory for data loaders."""

    # ----------- private attributes ----------- #
    _train_dataset: Optional[BaseDataset]
    """Optional[Dataset]: The training dataset."""
    _val_dataset: Optional[BaseDataset]
    """Optional[Dataset]: The validation dataset."""
    _test_dataset: Optional[BaseDataset]
    """Optional[Dataset]: The test dataset."""

    def __init__(
        self,
        batch_size: int,
        num_workers: Optional[int] = None,
        pin_memory: bool = False,
        train_dataset: Optional[BaseDataset] = None,
        valid_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
        *args,
        **kwargs,
    ) -> None:
        """Constructor function."""
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()
        self.pin_memory = pin_memory
        self._train_dataset = train_dataset
        self._val_dataset = valid_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            return None

        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            follow_batch=self._train_dataset.data_type.follow_batch,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.valid_dataset is None:
            return None

        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            follow_batch=self._val_dataset.data_type.follow_batch,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            return None

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            follow_batch=self._test_dataset.data_type.follow_batch,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    @property
    def train_dataset(self) -> Optional[BaseDataset]:
        """Optional[Dataset]: The training dataset or `None`."""
        return self._train_dataset

    @property
    def valid_dataset(self) -> Optional[BaseDataset]:
        """Optional[Dataset]: The validation dataset or `None`."""
        return self._val_dataset

    @property
    def test_dataset(self) -> Optional[BaseDataset]:
        """Optional[Dataset]: The test dataset or `None`."""
        return self._test_dataset
