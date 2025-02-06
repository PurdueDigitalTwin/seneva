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
"""Subsampler for the INTERACTION dataset."""
from dataclasses import dataclass
from typing import List, Optional

from interaction.dataset import LOCATIONS


@dataclass(frozen=True)
class INTERACTIONSubsampler:
    """A dataclass storing configs for filtering INTERACTION dataset scenarios.

    Attributes:
        ratio (float): A float subsample ratio between 0 and 1.
        locations (List[str]): A list of locations to include. If `None`, all
            locations in the INTERACTION dataset will be included.
    """

    ratio: float = 1.0
    locations: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert (
            isinstance(self.ratio, float)
            and self.ratio > 0.0
            and self.ratio <= 1.0
        ), ValueError(f"Invalid subsample ratio {self.ratio}")

        if self.locations is None:
            super().__setattr__("locations", LOCATIONS)
