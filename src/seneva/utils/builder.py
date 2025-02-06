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
"""Helper functions for building the project."""
from typing import List, Optional

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from seneva.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)


def build_callbacks(cfg: Optional[DictConfig] = None) -> List[Callback]:
    """Builds callback modules from configuration.

    Args:
        cfg (Optional[DictConfig], optional): The callback configurations.

    Returns:
        List[Callback]: The list of callback modules.
    """
    callbacks: List[Callback] = []

    if cfg is None:
        LOGGER.info("No callbacks specified. Skipping...")
        return callbacks

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Expect `cfg` to be a `DictConfig` object, "
            f"but got {type(cfg).__name__}."
        )

    for _, callback_cfg in cfg.items():
        # only instantiate if the config is a DictConfig and has `_target_` key
        if isinstance(callback_cfg, DictConfig) and "_target_" in callback_cfg:
            LOGGER.info(f"Building callback <{callback_cfg._target_}>...")
            callbacks.append(hydra.utils.instantiate(callback_cfg))
            LOGGER.info(f"Building callback <{callback_cfg._target_}>...DONE!")

    return callbacks


def build_loggers(cfg: Optional[DictConfig] = None) -> List[Logger]:
    """Builds logger modules from configuration.

    Args:
        cfg (Optional[DictConfig], optional): The logger configurations.

    Returns:
        List[Logger]: The list of logger modules.
    """
    loggers: List[Logger] = []

    if cfg is None:
        LOGGER.info("No loggers specified. Skipping...")
        return loggers

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Expect `cfg` to be a `DictConfig` object, "
            f"but got {type(cfg).__name__}."
        )

    for _, logger_cfg in cfg.items():
        # only instantiate if the config is a DictConfig and has `_target_` key
        if isinstance(logger_cfg, DictConfig) and "_target_" in logger_cfg:
            LOGGER.info(f"Building logger <{logger_cfg._target_}>...")
            loggers.append(hydra.utils.instantiate(logger_cfg))
            LOGGER.info(f"Building logger <{logger_cfg._target_}>...DONE!")

    return loggers
