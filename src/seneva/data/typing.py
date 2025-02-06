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
"""Common type definitions for the project data."""
from enum import Enum
from typing import Iterable, List, Union

from torch_geometric.transforms import BaseTransform

# Type aliases
Transforms = Union[BaseTransform, Iterable[BaseTransform]]


class MapPolylineTypes(Enum):
    """An enumeration of valid map polyline types.

    .. note::

        Without losing generalization, the common types of map polylines across
        different motion prediction algorithms consists of:
            * Lane centerlines -  which help navigation and path planning
            * Lane boundaries - which help define the drivable area
            * Stop lines - the stopping line at an intersection
            * Dashed white lane markings - separating two same-direction lanes
            * Solid white lane markings - separating two same-direction lanes,
                and also indicating the edge of the road
            * Dashed yellow lane markings - separating two opposite-direction
                lanes, but passing is allowed
            * Solid yellow lane markings - separating two opposite-direction
                lanes, and also indicating the edge of the road
            * Crosswalks - the boundary of a pedestrian crossing area
            * Speed bumps - the boundary of a speed bump area
            * Driveways - the boundary of a driveway area
    """

    UNKNOWN = 0
    """int: A placeholder for unknown map polyline type."""
    LANE_CENTERLINE = 1
    """int: The lane centerline type."""
    LANE_BOUNDARY = 2
    """int: The lane boundary type."""
    STOP_LINE = 3
    """int: The stop line type."""
    DASHED_WHITE_LANE_MARKING = 4
    """int: The dashed white lane marking type."""
    SOLID_WHITE_LANE_MARKING = 5
    """int: The solid white lane marking type."""
    DASHED_YELLOW_LANE_MARKING = 6
    """int: The dashed yellow lane marking type."""
    SOLID_YELLOW_LANE_MARKING = 7
    """int: The solid yellow lane marking type."""
    CROSSWALK = 8
    """int: The crosswalk type."""
    SPEED_BUMP = 9
    """int: The speed bump type."""
    DRIVEWAY = 10
    """int: The driveway type."""

    @property
    def one_hot_encoding(self) -> List[int]:
        """List[int]: The one-hot encoding of the map polyline type."""
        return [int(self == t) for t in MapPolylineTypes]

    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


class AgentTypes(Enum):
    """An enumeration of valid agent types in the dataset.

    .. note::

        The common types of agents in the dataset include:
            * Vehicle - a car, truck, or bus
            * Pedestrian - a person walking
            * Cyclist - a person riding a bicycle
    """

    UNKNOWN = 0
    """int: A placeholder for unknown agent type."""
    VEHICLE = 1
    """int: The vehicle type."""
    PEDESTRIAN = 2
    """int: The pedestrian type."""
    CYCLIST = 3
    """int: The cyclist type."""

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"
