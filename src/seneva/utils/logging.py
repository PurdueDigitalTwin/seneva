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
"""Helper functions for logging."""
import logging
import os

from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from torch.nn import Module


def get_log_level() -> int:
    """Get the log level from environment variable.

    Returns:
        int: The log level.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging._nameToLevel.get(level, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level}")

    return level


def get_logger(name=__name__) -> logging.Logger:
    """Initialize the logger service.

    Args:
        name (str, optional): The name of the logger. Defaults to ``__name__``.

    Returns:
        logging.Logger: The logger object.

    Raises:
        ValueError: If the log level is invalid.
    """

    # get level
    logger = logging.getLogger(name)
    level = get_log_level()
    logging.root.setLevel(level)
    logger.setLevel(level)

    # use `rank_zero_only` to ensure that all logging levels get marked with
    # rank zero decorator, and avoid duplicate logging from each process.
    logging_levels = [
        name.lower()
        for name in logging._nameToLevel.keys()
        if name != "NOTSET"
    ]
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def log_hyperparams(params: dict) -> None:
    """Controls which config parts are to be logged by lightning loggers.

    Args:
        params (dict): The config dictionary to be logged.
    """
    logger = get_logger()
    hparams = {}

    cfg = params["cfg"]
    assert isinstance(cfg, DictConfig)
    model = params["model"]
    assert isinstance(model, (Module, LightningModule))
    trainer = params["trainer"]
    assert isinstance(trainer, Trainer)

    if not trainer.logger:
        logger.info("Logger not found! Skipping hyperparameter logging...")
        return

    # log the number of model parameters
    hparams["model"] = cfg["model"]
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # log configurations
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["checkpoint"] = cfg.get("checkpoint")
    hparams["data"] = cfg["data"]
    hparams["extras"] = cfg.get("extras")
    hparams["task"] = cfg.get("task")
    hparams["seed"] = cfg.get("seed")
    hparams["tags"] = cfg.get("tags")
    hparams["trainer"] = cfg["trainer"]

    # log the hyperparameters
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
