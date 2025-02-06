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
"""Dataset API for the UC Berkeley INTERACTION dataset."""
import gzip
import pickle
import random
import traceback
from collections.abc import Sequence
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.map.lane_segment import LaneMarkType  # type: ignore
from av2.map.map_api import ArgoverseStaticMap  # type: ignore
from interaction.dataset import SPLITS
from interaction.dataset.map_api import INTERACTIONMap
from interaction.dataset.track_api import INTERACTIONCase, INTERACTIONScenario
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from torch import Tensor
from torch_geometric.transforms import Compose

from seneva.data.base import BaseDataset, PolylineData
from seneva.data.subsampler import INTERACTIONSubsampler
from seneva.data.typing import Transforms
from seneva.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)
WAY_TYPE_OF_INTEREST = (
    "CURBSTONE_LOW",
    "GUARD_RAIL",
    "ROAD_BORDER",
    "LINE_THIN_SOLID",
    "LINE_THIN_SOLID_SOLID",
    "LINE_THIN_DASHED",
    "LINE_THICK_SOLID",
    "LINE_THICK_SOLID_SOLID",
    "STOP_LINE",
    "PEDESTRIAN_MARKING",
    "VIRTUAL",
    "VIRTUAL_SOLID",
)


class Argoverse2Data(PolylineData):
    """Implementation of the data class for the Argoverse 2 dataset."""

    @property
    def map_feature_dims(self) -> int:
        return 2 + (len(LaneMarkType) + 1)  # NOTE: xy, type_oh

    @property
    def motion_feature_dims(self) -> int:
        return 5 + len(ObjectType)  # NOTE: xy, heading, vx, vy, type_oh


class Argoverse2Dataset(BaseDataset):
    """Dataset class of the Argoverse 2 dataset."""

    split: str
    """str: The split of the dataset, either `train`, `val`, or `test`."""
    radius: Optional[float]
    """Optional[float]: The radius of the observation buffer."""

    _scenario_ids: List[str]
    """List[str]: The list of scenario IDs in the dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Transforms] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Constructor function.

        Args:
            root (str): The root directory of the dataset.
            split (str, optional): The split of the dataset, either `train`,
                `val`, or `test`. Defaults to `train`.
            transform (Optional[Transforms], optional): The transformation
                modules applied to the dataset. Defaults to `None`.
            radius (Optional[float], optinoal): The radius of the observation
                buffer. Defaults to `None`.
        """
        assert Path(root).exists(), f"Invalid data root: {root}"
        assert split in ("train", "val", "test"), (
            "Expect split to be either 'train', 'val', or 'test', "
            f"but got {split:s}."
        )
        self.split = split
        self.radius = radius
        # Initialize the container for scenario IDs
        self._scenario_ids = [
            subdir.name for subdir in Path(root, "raw", self.split).iterdir()
        ]
        LOGGER.info(f"Found {len(self._scenario_ids):d} {split:s} scenarios.")

        if isinstance(transform, Sequence):
            transform = Compose(transform)

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=None,
            pre_filter=None,
        )

    @property
    def data_type(self) -> Type[Argoverse2Data]:
        return Argoverse2Data

    @property
    def raw_file_names(self) -> List[str]:
        base_dir = Path(self.raw_dir, self.split)
        map_files = [
            base_dir.joinpath(idx, self._map_file_name(idx)).resolve()
            for idx in self._scenario_ids
        ]
        scenario_files = [
            base_dir.joinpath(idx, self._scenario_file_name(idx)).resolve()
            for idx in self._scenario_ids
        ]
        return map_files + scenario_files

    def download(self) -> None:
        """Download the Argoverse 2 dataset.

        Raises:
            RuntimeError: If the raw data files are not found or incomplete.
        """
        raise RuntimeError(
            "Raw data files not found or incomplete at "
            f"{self.raw_dir:s}. Please refer to the README for "
            "instructions on how to download and preparare the dataset."
        )

    def get(self, idx: int) -> Argoverse2Data:
        scenario_id = self._scenario_ids[idx]
        map_file, scenario_file = self.get_scenario_files(scenario_id)
        map_api = ArgoverseStaticMap.from_json(map_file)
        scenario_api = load_argoverse_scenario_parquet(scenario_file)

        # parse motion features
        track_data = self._parse_track_data(scenario_api=scenario_api)
        anchor = tuple(track_data["anchor"][0:2])

        # parse map features
        map_data = self._parse_map_data(map_api=map_api, anchor=anchor)

        data = Argoverse2Data(
            x_map=map_data["x_map"],
            map_cluster=map_data["map_cluster"],
            map_pos=map_data["map_pos"],
            x_motion=track_data["x_motion"],
            motion_cluster=track_data["motion_cluster"],
            motion_timestamps=track_data["motion_timestamps"],
            track_pos=track_data["track_pos"],
            track_ids=track_data["track_ids"],
            y_motion=track_data["motion_tar"],
            y_cluster=track_data["motion_tar_cluster"],
            y_valid=track_data["motion_tar_valid"],
            anchor=track_data["anchor"],
            num_agents=track_data["num_agents"],
            tracks_to_predict=track_data["tracks_to_predict"],
        )
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        assert data.is_valid(), "Invalid data generated!"

        return data

    def len(self) -> int:
        return len(self._scenario_ids)

    def get_scenario_files(self, scenario_id: str) -> Tuple[Path, Path]:
        map_dir = Path(
            self.raw_dir,
            self.split,
            scenario_id,
            self._map_file_name(scenario_id),
        ).resolve()
        scenario_dir = Path(
            self.raw_dir,
            self.split,
            scenario_id,
            self._scenario_file_name(scenario_id),
        ).resolve()

        return map_dir, scenario_dir

    def _parse_map_data(
        self,
        map_api: ArgoverseStaticMap,
        anchor: Tuple[float, float],
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(map_api, ArgoverseStaticMap), "Invalid map api!"

        if self.radius is not None and isinstance(self.radius, (int, float)):
            if self.radius == 0.0:
                # NOTE: An empty buffer results in empty map data
                return {
                    "x_map": torch.zeros(
                        (0, Argoverse2Data.map_feature_dims),
                        dtype=torch.float32,
                    ),
                    "map_cluster": torch.zeros((0,), dtype=torch.long),
                    "map_pos": torch.zeros((0, 2), dtype=torch.float32),
                }
            if self.radius > 0:
                lanes = map_api.get_nearby_lane_segments(
                    query_center=np.array(anchor[0:2], dtype=np.float64),
                    search_radius_m=float(self.radius),
                )
                # NOTE: replace with `map_api.get_nearby_ped_crossings`
                # after it is implemented by the Argoverse team
                # crosswalks = map_api.get_nearby_ped_crossings(
                #     query_center=np.array(anchor[0:2], dtype=np.float64),
                #     search_radius_m=float(self.radius),
                # )
                buffer: Polygon = Point(anchor).buffer(self.radius)
                crosswalks = [
                    cw
                    for cw in map_api.vector_pedestrian_crossings.values()
                    if buffer.intersects(
                        Polygon(
                            shell=np.concatenate(
                                [
                                    cw.edge1.xyz[:, 0:2],
                                    cw.edge2.xyz[::-1, 0:2],
                                ],
                                axis=0,
                            )
                        )
                    )
                ]
        else:
            lanes = map_api.vector_lane_segments.values()
            crosswalks = map_api.vector_pedestrian_crossings.values()

        x_map, map_cluster, map_pos = [], [], []
        num_lines: int = 0
        # parse lane boundary polyline features
        for lane in lanes:
            for mark in (lane.left_lane_marking, lane.right_lane_marking):
                type_oh = [0] * (len(LaneMarkType) + 1)
                if mark.mark_type in (LaneMarkType.UNKNOWN, LaneMarkType.NONE):
                    if lane.is_intersection:
                        type_oh[-2] = 1  # NOTE: virtual boundary
                    else:
                        type_oh[-3] = 1  # NOTE: solid boundary
                else:
                    idx = list(LaneMarkType).index(mark.mark_type)
                    type_oh[idx] = 1
                cluster = np.array([num_lines] * mark.polyline.shape[0])
                x_map.append(
                    np.concatenate(
                        [
                            mark.polyline[:, 0:2],
                            np.array(type_oh)[None, :].repeat(
                                mark.polyline.shape[0], axis=0
                            ),
                        ],
                        axis=1,
                    )
                )
                map_cluster.append(cluster)
                map_pos.append(
                    mark.polyline[:, 0:2].mean(axis=0, keepdims=True)
                )
                num_lines += 1

        # parse crosswalk polyline features
        for crosswalk in crosswalks:
            for edge in (crosswalk.edge1, crosswalk.edge2):
                type_oh = [0] * (len(LaneMarkType) + 1)
                type_oh[-1] = 1
                cluster = np.array([num_lines] * edge.xyz.shape[0])
                x_map.append(
                    np.concatenate(
                        [
                            edge.xyz[:, 0:2],
                            np.array(type_oh)[None, :].repeat(
                                edge.xyz.shape[0], axis=0
                            ),
                        ],
                        axis=1,
                    )
                )
                map_cluster.append(cluster)
                map_pos.append(edge.xyz[:, 0:2].mean(axis=0, keepdims=True))
                num_lines += 1

        x_map = np.concatenate(x_map, axis=0)
        map_cluster = np.concatenate(map_cluster, axis=0)
        map_pos = np.concatenate(map_pos, axis=0)

        return {
            "x_map": torch.from_numpy(x_map).float(),
            "map_cluster": torch.from_numpy(map_cluster).long(),
            "map_pos": torch.from_numpy(map_pos).float(),
        }

    def _parse_track_data(
        self, scenario_api: ArgoverseScenario
    ) -> Dict[str, torch.Tensor]:
        assert isinstance(scenario_api, ArgoverseScenario), "Invalid scenario!"
        # parse track motion features
        x_motion, motion_cluster, motion_timestamps = [], [], []
        track_pos, track_ids = [], []
        num_tracks: int = 0

        if self.radius is not None and self.radius == 0.0:
            return {
                "x_motion": torch.zeros(
                    (0, Argoverse2Data.motion_feature_dims),
                    dtype=torch.float32,
                ),
                "motion_cluster": torch.zeros((0,), dtype=torch.long),
                "motion_timestamps": torch.zeros((0,), dtype=torch.long),
                "track_pos": torch.zeros((0, 2), dtype=torch.float32),
                "track_ids": torch.zeros((0,), dtype=torch.long),
                "motion_tar": torch.zeros((0, 5), dtype=torch.float32),
                "motion_tar_cluster": torch.zeros((0,), dtype=torch.long),
                "motion_tar_valid": torch.zeros((0, 5), dtype=torch.bool),
                "anchor": torch.zeros((3,), dtype=torch.float32),
                "num_agents": torch.zeros((1,), dtype=torch.long),
                "tracks_to_predict": torch.zeros((0,), dtype=torch.long),
            }

        # parse anchor of the scenario
        for track in scenario_api.tracks:
            if track.category == TrackCategory.FOCAL_TRACK:
                for i, state in enumerate(track.object_states):
                    next_state = track.object_states[i + 1]
                    if state.observed is True and next_state.observed is False:
                        anchor = np.array(
                            [
                                state.position[0],
                                state.position[1],
                                state.heading,
                            ]
                        )
                        break
                break

        # initialize target feature containers
        if self.split not in ("train", "val"):
            # for test set, there are no target motion states
            motion_tar = torch.zeros((0, 5), dtype=torch.float32)
            motion_tar_cluster = torch.zeros((0, 5), dtype=torch.long)
            motion_tar_valid = torch.zeros((0, 5), dtype=torch.bool)

        for track in scenario_api.tracks:
            states = [
                np.array(
                    (
                        *state.position,
                        state.heading,
                        *state.velocity,
                        state.timestep,
                    ),
                )[None, :]
                for state in track.object_states
                if state.observed is True
            ]
            if len(states) == 0:
                continue
            states = np.concatenate(states, axis=0)
            type_oh = np.zeros(shape=(len(states), len(ObjectType)))
            idx = list(ObjectType).index(track.object_type)
            type_oh[:, idx] = 1
            cluster = np.array([num_tracks] * len(states))
            timesteps = states[:, -1]

            # Filter out the states that are not in the observation buffer
            if self.radius is not None and self.radius > 0.0:
                mask: npt.NDArray[np.bool_] = np.less_equal(
                    np.linalg.norm(
                        states[:, 0:2] - anchor[None, 0:2],
                        axis=1,
                        ord=2,
                    ),
                    self.radius,
                )
                if mask.sum() == 0:
                    continue
                else:
                    states = states[mask]
                    cluster = cluster[mask]
                    timesteps = timesteps[mask]
                    type_oh = type_oh[mask]

            x_motion.append(np.concatenate([states[:, :-1], type_oh], axis=1))
            motion_cluster.append(cluster)
            motion_timestamps.append(timesteps)
            track_pos.append(states[-1:, 0:2])
            track_ids.append(num_tracks + 1)
            if track.category == TrackCategory.FOCAL_TRACK:
                tracks_to_predict = torch.tensor([num_tracks + 1]).long()
                if self.split in ("train", "val"):
                    # for training and validation sets, parse the target states
                    motion_tar = torch.from_numpy(
                        np.concatenate(
                            [
                                np.array(
                                    (
                                        *state.position,
                                        state.heading,
                                        *state.velocity,
                                    )
                                )[None, ...]
                                for state in track.object_states
                                if state.timestep >= 50
                            ]
                        )
                    )
                    motion_tar_cluster = torch.tensor(
                        [num_tracks + 1] * len(motion_tar), dtype=torch.long
                    )
                    motion_tar_valid = torch.ones_like(motion_tar).bool()
            num_tracks += 1
        x_motion = np.concatenate(x_motion, axis=0)
        motion_cluster = np.concatenate(motion_cluster, axis=0)
        motion_timestamps = np.concatenate(motion_timestamps, axis=0)
        track_pos = np.concatenate(track_pos, axis=0)
        track_ids = np.array(track_ids)

        return {
            "x_motion": torch.from_numpy(x_motion).float(),
            "motion_cluster": torch.from_numpy(motion_cluster).long(),
            "motion_timestamps": torch.from_numpy(motion_timestamps).long(),
            "track_pos": torch.from_numpy(track_pos).float(),
            "track_ids": torch.from_numpy(track_ids).long(),
            "motion_tar": motion_tar,
            "motion_tar_cluster": motion_tar_cluster,
            "motion_tar_valid": motion_tar_valid,
            "anchor": torch.from_numpy(anchor).float(),
            "num_agents": torch.tensor([num_tracks]).long(),
            "tracks_to_predict": tracks_to_predict,
        }

    def _map_file_name(self, scenario_id: str) -> str:
        """Returns the raw map file name with the given scenario ID."""
        return f"log_map_archive_{scenario_id}.json"

    def _scenario_file_name(self, scenario_id: str) -> str:
        """Returns the raw scenario file name with the given scenario ID."""
        return f"scenario_{scenario_id}.parquet"


class INTERACTIONData(PolylineData):
    """Implementation of the data class for the INTERACTION dataset."""

    @property
    def map_feature_dims(self) -> int:
        return 9  # NOTE: xy, type_oh

    @property
    def motion_feature_dims(self) -> int:
        return 7  # NOTE: xy, heading, vx, vy, length, width


class INTERACTIONDataset(BaseDataset):
    """Dataset class of the INTERACTION dataset."""

    challenge_type: str
    """str: name of the challenge, either `single-agent`,
        `conditional-single-agent`, `multi-agent`, or
        `conditional-multi-agent`."""
    split: str
    """str: tag of the dataset, either `train`, `val`, or `test`."""
    subsampler: INTERACTIONSubsampler
    """INTERACTIONSubsampler: subsampler for subsampling."""
    radius: float
    """float: query range in meters."""
    train_on_multi_agent: bool
    """bool: if train on multi-agent prediction."""

    # private attributes
    _indexer: List[Tuple[str, str, str]]
    """List[Tuple[str, str, str]]: A list of scenario indexers."""
    _map_api_container: Dict[str, INTERACTIONMap]
    """Dict[str, INTERACTIONMap]: map API container."""
    _track_api_container: Dict[str, INTERACTIONScenario]
    """Dict[str, INTERACTIONScenario]: track API container."""

    def __init__(
        self,
        root: str,
        challenge_type: str,
        subsampler: INTERACTIONSubsampler,
        split: str = "train",
        transform: Optional[Transforms] = None,
        radius: float = 50,
        train_on_multi_agent: bool = False,
        num_workers: int = 0,
    ) -> None:
        """Constructor function.

        Args:
            root (str): root directory of the INTERACTION dataset files.
            challenge_type (str): name of the challenge, either `single-agent`,
                `conditional-single-agent`, `multi-agent`, or
                `conditional-multi-agent`.
            subsampler (INTERACTIONSubsampler): subsampler for subsampling.
            split (str): tag of the dataset, either `train`, `val`, or `test`.
            transform (Optional[Transforms], optional): data transform modules.
                Defaults to `None`.
            radius (float, optional): query range in meters. Defaults to 50.
            num_workers (int): number of workers for multiprocessing.
                Defaults objto :math:`0`.
        """
        assert challenge_type in [
            "single-agent",
            "multi-agent",
            "conditional-single-agent",
            "conditional-multi-agent",
        ], ValueError(f"Invalid challenge type {challenge_type:s}.")
        self.challenge_type = challenge_type

        assert split in ["train", "trainval", "val", "test"], ValueError(
            "Expect tag to be either 'train', 'trainval', 'val', or 'test', "
            f"but got {split:s}."
        )
        self.split = split
        self.subsampler = subsampler
        self.radius = radius
        self.train_on_multi_agent = train_on_multi_agent

        self._map_api_container, self._track_api_container = {}, {}
        self._indexer = []

        if isinstance(num_workers, int) and num_workers > 0:
            self._num_workers = num_workers
        else:
            self._num_workers = cpu_count()

        if isinstance(transform, Sequence):
            transform = Compose(list(transform))

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=None,
            pre_filter=None,
        )

        self._load_to_mem()

    @property
    def data_type(self) -> Type[INTERACTIONData]:
        """Type[INTERACTIONData]: The type of data class to use."""
        return INTERACTIONData

    @property
    def indexer(self) -> List[Tuple[str, str, str]]:
        """List[Tuple[str, str, str]]: A list of scenario indexers."""
        return self._indexer

    @property
    def raw_file_names(self) -> List[str]:
        """List[str]: A list of raw file names."""
        return (
            self._get_raw_map_file_paths() + self._get_raw_track_file_paths()
        )

    @property
    def processed_file_names(self) -> List[str]:
        """List[str]: A list of processed file names."""
        return (
            self._get_processed_map_file_paths()
            + self._get_processed_track_file_paths()
        )

    @property
    def cache_dir(self) -> str:
        """str: absolute path of the cache directory."""
        return str(
            Path(self.processed_dir, "cache", self.challenge_type).resolve()
        )

    @property
    def map_root(self) -> str:
        """str: absolute root directory of all the map data files."""
        return str(Path(self.raw_dir, "maps").resolve())

    @property
    def track_root(self) -> str:
        """str: absolute root directory of all the track data files."""
        return str(Path(self.raw_dir, self.tag).resolve())

    @property
    def locations(self) -> List[str]:
        """List[str]: A list of locations in the dataset."""
        if self.split == "trainval":
            tag = "train"
        else:
            tag = self.tag

        return [
            name for name in SPLITS[tag] if name in self.subsampler.locations
        ]

    @property
    def tag(self) -> str:
        """str: Tag of the dataset. See official website for details."""
        if self.split == "test":
            return f"{self.split}_{self.challenge_type}"
        elif self.split == "trainval":
            return "train"
        return self.split

    def download(self) -> None:
        """Download the INTERACTION dataset to the `self.raw_dir` folder.

        Raises:
            RuntimeError: if the raw data files are not found or incomplete.
        """
        raise RuntimeError(
            "Raw data files not found or incomplete at"
            f" '{self.raw_dir:s}'! "
            "Please visit 'https: // interaction-dataset.com', "
            "download and validate all the data files."
        )

    def get(self, idx: int) -> PolylineData:
        """Get the data object at the given index.

        Args:
            idx (int): index of the data object.

        Returns:
            PolylineData: the data object at the given index.
        """
        location, case_id, ego_id = self._indexer[idx]
        map_api = self._map_api_container[location]
        case_api = self._track_api_container[location][case_id]

        data = self._get_case_data(
            map_api=map_api, case_api=case_api, ego_id=ego_id
        )
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        assert data.is_valid(), "Invalid data generated!"

        return data

    def len(self) -> int:
        return len(self._indexer)

    def process(self):
        queue = Queue()
        processes: List[Process] = []
        for location in self.locations:
            proc = Process(
                target=self._parse_api,
                args=(
                    queue,
                    location,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

        if not queue.empty():
            error_, traceback_ = queue.get()
            raise RuntimeError(error_)

    def _get_raw_map_file_paths(self) -> List[Path]:
        """Return a list of absolute paths of the raw map data files."""
        return [
            Path(self.map_root, f"{location}.osm").resolve()
            for location in self.locations
        ]

    def _get_processed_map_file_path(self, location: str) -> Path:
        """Return the absolute path of the processed map data file."""
        map_dir = Path(self.processed_dir, "maps")
        if not map_dir.is_dir():
            map_dir.mkdir(parents=True, exist_ok=True)

        return Path(map_dir, f"{location}.gz").resolve()

    def _get_processed_map_file_paths(self) -> List[Path]:
        """Return a list of absolute paths of the processed map data files."""
        return [
            self._get_processed_map_file_path(location)
            for location in self.locations
        ]

    def _get_raw_track_file_paths(self) -> List[Path]:
        if "test" in self.tag:
            suffix = "obs"
        else:
            suffix = self.tag

        return [
            Path(self.track_root, f"{location}_{suffix}.csv").resolve()
            for location in self.locations
        ]

    def _get_processed_track_file_path(self, location: str) -> Path:
        track_dir = Path(self.processed_dir, "tracks")
        if not track_dir.is_dir():
            track_dir.mkdir(parents=True, exist_ok=True)

        return Path(track_dir, f"{location}_{self.split}.gz").resolve()

    def _get_processed_track_file_paths(self) -> List[Path]:
        return [
            self._get_processed_track_file_path(location)
            for location in self.locations
        ]

    def _get_case_map(
        self, map_api: INTERACTIONMap, anchor: Tuple[float, float]
    ) -> Dict[str, Tensor]:
        assert isinstance(map_api, INTERACTIONMap), "Invalid map api."

        def _parse_map_features(record: pd.Series) -> Tuple[npt.NDArray, ...]:
            obj_id = record.name

            obj_geom: BaseGeometry = record["geometry"]
            if isinstance(obj_geom, Point):
                # single-point polyline
                x, y = obj_geom.xy[0][0], obj_geom.xy[1][0]
                feats = np.array([x, y], dtype=np.float64)
            elif isinstance(obj_geom, LineString):
                feats = np.vstack(obj_geom.coords)
            elif isinstance(obj_geom, MultiLineString):
                xs, ys = [], []
                for line in obj_geom.geoms:
                    xs.extend(line.xy[0])
                    ys.extend(line.xy[1])
                feats = np.vstack([xs, ys]).T
            else:
                raise NotImplementedError(
                    f"Unsupported geometry type: {obj_geom.type:s}."
                )
            try:
                centroid = np.mean(feats[:, 0:2], axis=0)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get centroid of geometry {obj_geom}: {e}"
                )

            # encode map polyline type
            if record["type"] in (
                "CURBSTONE_LOW",
                "GUARD_RAIL",
                "ROAD_BORDER",
            ):
                type_encoding = np.array([1, 0, 0, 0, 0, 0, 0])
            elif record["type"] == "LINE_THIN_DASHED":
                type_encoding = np.array([0, 1, 0, 0, 0, 0, 0])
            elif record["type"] in (
                "LINE_THIN_SOLID",
                "LINE_THIN_SOLID_SOLID",
            ):
                type_encoding = np.array([0, 0, 1, 0, 0, 0, 0])
            elif record["type"] in (
                "LINE_THICK_SOLID",
                "LINE_THICK_SOLID_SOLID",
            ):
                type_encoding = np.array([0, 0, 0, 1, 0, 0, 0])
            elif record["type"] == "STOP_LINE":
                type_encoding = np.array([0, 0, 0, 0, 1, 0, 0])
            elif record["type"] == "PEDESTRIAN_MARKING":
                type_encoding = np.array([0, 0, 0, 0, 0, 1, 0])
            elif record["type"] in ("VIRTUAL", "VIRTUAL_SOLID"):
                type_encoding = np.array([0, 0, 0, 0, 0, 0, 1])
            type_encoding = np.broadcast_to(
                type_encoding[None, :],
                (feats.shape[0], len(type_encoding)),
            )

            feats = np.hstack(
                (
                    feats,
                    type_encoding,
                    np.full((feats.shape[0], 1), fill_value=obj_id),
                )
            )

            return feats, centroid

        # parse node features from map layers
        waylayer = map_api.get_map_layer("way")
        waylayer = waylayer[waylayer["type"].isin(WAY_TYPE_OF_INTEREST)]
        # optionally create observation range buffer and clip observations
        if self.radius is not None:
            buffer: Polygon = Point(anchor).buffer(self.radius)
            waylayer = gpd.clip(waylayer, buffer)
        # remap polyline ids
        map_id_mapper = {
            _id: ind
            for ind, _id in enumerate(waylayer.index.unique().tolist())
        }
        waylayer.index = waylayer.index.map(map_id_mapper)

        # parse node features
        node_feats = waylayer.apply(_parse_map_features, axis=1)
        map_feat = np.vstack([out[0] for out in node_feats])
        map_pos = np.vstack([out[1] for out in node_feats])

        x_map = torch.from_numpy(map_feat[:, :-1]).float()
        map_cluster = torch.from_numpy(map_feat[:, -1]).long()
        map_pos = torch.from_numpy(map_pos).float()

        return {"x_map": x_map, "map_cluster": map_cluster, "map_pos": map_pos}

    def _get_case_tracks(
        self,
        case_api: INTERACTIONCase,
        tracks_to_predict: Sequence[int],
        anchor: Tuple[float, float],
    ) -> Dict[str, Tensor]:
        """Get track data from case api.

        Args:
            case_api (INTERACTIONCase): The case data API object.
            anchor (Tuple[float, float]): The anchor point of the map.

        Returns:
            Dict[str, Tensor]: track data
        """
        hist_df = case_api.history_frame.fillna(0.0)

        # optionally create observation range buffer and clip observations
        if self.radius is not None:
            pos = torch.from_numpy(hist_df.loc[:, ["x", "y"]].values).float()
            dist = torch.cdist(
                x1=pos, x2=torch.tensor(anchor[0:2]).view(1, 2).float(), p=2
            )
            hist_df = hist_df.loc[dist.cpu().numpy() <= self.radius]

        # NOTE: change the following lines for different motion features
        cols = ["x", "y", "psi_rad", "vx", "vy", "length", "width"]
        track_ids = torch.from_numpy(hist_df.index.unique().values).long()
        track_pos = torch.from_numpy(
            hist_df.reset_index(drop=False)
            .groupby("track_id")
            .agg({"x": "last", "y": "last"})
            .values
        ).float()

        motion_clusterss = (
            hist_df.index.to_series()
            .apply(lambda x: track_ids.tolist().index(x))
            .values
        )
        motion_data = torch.from_numpy(hist_df.loc[:, cols].values).float()
        motion_cluster = torch.from_numpy(motion_clusterss).long()
        motion_timestamps = torch.from_numpy(
            hist_df.loc[:, "timestamp_ms"].values
        ).long()

        # get target motion states
        future_df = case_api.futural_frame
        future_df = future_df.loc[future_df.index.isin(tracks_to_predict)]
        if len(future_df) > 0:
            # for training and validation set, there are target motion states
            motion_tar = torch.from_numpy(
                future_df.loc[:, ["x", "y", "psi_rad", "vx", "vy"]].values
            ).float()
            motion_tar_cluster = torch.from_numpy(
                future_df.index.values
            ).long()
            motion_tar_valid = ~torch.from_numpy(
                future_df.loc[:, cols].isna().values
            ).bool()
        else:
            # for test set, there are no target motion states
            motion_tar = torch.zeros((0, len(cols)), dtype=torch.float32)
            motion_tar_cluster = torch.zeros((0,), dtype=torch.long)
            motion_tar_valid = torch.zeros((0, len(cols)), dtype=torch.bool)

        return {
            "motion_data": motion_data,
            "motion_cluster": motion_cluster,
            "motion_timestamps": motion_timestamps,
            "track_ids": track_ids,
            "track_pos": track_pos,
            "motion_tar": motion_tar,
            "motion_tar_valid": motion_tar_valid,
            "motion_tar_cluster": motion_tar_cluster,
        }

    def _get_case_data(
        self,
        map_api: INTERACTIONMap,
        case_api: INTERACTIONCase,
        ego_id: Union[int, Sequence[int]],
    ) -> INTERACTIONData:
        if self.split == "train" and self.train_on_multi_agent:
            tracks_to_predict = case_api.tracks_to_predict
        elif isinstance(ego_id, Sequence):
            tracks_to_predict = list(ego_id)
        elif isinstance(ego_id, int):
            tracks_to_predict = [ego_id]
        else:
            raise RuntimeError(f"Invalid ego_id: {ego_id}.")

        if isinstance(ego_id, Sequence):
            anchor = case_api.current_frame.loc[
                case_api.current_frame.index.isin(ego_id),
                ["x", "y", "psi_rad"],
            ].values.mean(0)
        else:
            anchor = case_api.current_frame.loc[ego_id, ["x", "y", "psi_rad"]]
            anchor = anchor.values.astype(np.float64)

        # process track data
        tracks_data = self._get_case_tracks(
            case_api=case_api,
            tracks_to_predict=tracks_to_predict,
            anchor=tuple(anchor[0:2]),
        )

        # process map data
        map_data = self._get_case_map(map_api, tuple(anchor))

        # process system information
        anchor = torch.tensor(anchor, dtype=torch.float32)
        tracks_to_predict = torch.tensor(tracks_to_predict, dtype=torch.long)

        return INTERACTIONData(
            x_map=map_data["x_map"],
            map_pos=map_data["map_pos"],
            map_cluster=map_data["map_cluster"],
            x_motion=tracks_data["motion_data"],
            motion_cluster=tracks_data["motion_cluster"],
            motion_timestamps=tracks_data["motion_timestamps"],
            track_pos=tracks_data["track_pos"],
            track_ids=tracks_data["track_ids"],
            y_motion=tracks_data["motion_tar"],
            y_valid=tracks_data["motion_tar_valid"],
            y_cluster=tracks_data["motion_tar_cluster"],
            anchor=anchor,
            tracks_to_predict=tracks_to_predict,
            num_agents=torch.tensor([case_api.num_agents], dtype=torch.long),
        )

    def _load_to_mem(self) -> None:
        """Load all intermediate API objects to memory."""
        LOGGER.info(f"Loading {self.tag} scenarios from cache...")
        for location in self.locations:
            with gzip.open(
                self._get_processed_map_file_path(location), "rb"
            ) as file:
                map_api: INTERACTIONMap = pickle.load(file)
            with gzip.open(
                self._get_processed_track_file_path(location), "rb"
            ) as file:
                track_api: INTERACTIONScenario = pickle.load(file)

            self._map_api_container[location] = map_api
            self._track_api_container[location] = track_api

            if self.subsampler.ratio < 1.0:
                k = track_api.num_cases * self.subsampler.ratio
                self._indexer.extend(
                    # single-agent prediction
                    [
                        (track_api.location, case_id, track_id)
                        for (case_id, track_ids) in random.sample(
                            track_api._tracks_to_predict.items(), k=int(k)
                        )
                        for track_id in track_ids
                    ]
                    if "single" in self.challenge_type
                    # multi-agent prediction
                    else [
                        (track_api.location, case_id, track_ids)
                        for (case_id, track_ids) in random.sample(
                            track_api._tracks_to_predict.items(), k=int(k)
                        )
                    ]
                )

            else:
                self._indexer.extend(
                    # single-agent prediction
                    [
                        (track_api.location, case_id, track_id)
                        for (
                            case_id,
                            track_ids,
                        ) in track_api._tracks_to_predict.items()
                        for track_id in track_ids
                    ]
                    if "single" in self.challenge_type
                    # multi-agent prediction
                    else [
                        (track_api.location, case_id, track_ids)
                        for (
                            case_id,
                            track_ids,
                        ) in track_api._tracks_to_predict.items()
                    ]
                )

        LOGGER.info("Loading scenarios for dataset...DONE!")

    def _parse_api(self, queue: Queue, location: str) -> None:
        """Parse scenario APIs and save to cache."""
        try:
            LOGGER.info("Processing scenario %s...", location)
            map_filepath = self._get_processed_map_file_path(location)
            if not map_filepath.is_file():
                # process and cache map data
                map_api = INTERACTIONMap(root=self.map_root, location=location)
                with gzip.open(map_filepath, mode="wb") as file:
                    pickle.dump(map_api, file)

            track_filepath = self._get_processed_track_file_path(location)
            if not track_filepath.is_file():
                # process and cache scenario api
                track_api = INTERACTIONScenario(
                    root=self.track_root,
                    location=location,
                    split="train" if self.split == "trainval" else self.split,
                )
                with gzip.open(track_filepath, mode="wb") as file:
                    pickle.dump(track_api, file)
            LOGGER.info("Processing scenario %s...DONE!", location)
        except Exception as err:
            tb = traceback.format_exc()
            queue.put((err, tb))

    def __str__(self) -> str:
        attr_str = ", ".join(
            [
                f"challenge_type={self.challenge_type}",
                f"split={self.split}",
                f"num_data={self.len()}",
            ]
        )
        return (
            f"<{self.__class__.__name__}({attr_str})" f" at {hex(id(self))}>"
        )

    def __repr__(self) -> str:
        return str(self)
