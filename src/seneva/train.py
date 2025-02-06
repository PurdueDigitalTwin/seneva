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
"""Entry point script for training the model."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from seneva.utils.builder import build_callbacks, build_loggers
from seneva.utils.logging import get_logger, log_hyperparams
from seneva.utils.tools import apply_extras, get_metric_value, task_wrapper

# Constants
HYDRA_CFG = {
    "config_path": str(Path(__file__).parents[1].joinpath("config").resolve()),
    "config_name": "default_train.yaml",
    "version_base": "1.3",
}
LOGGER = get_logger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Initialize and train the model.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple of two dictionary
            of metrics and objects, respectively.
    """

    # set global seed for reproducibility if cfg.get("seed"):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # build callbacks
    LOGGER.info("Building callbacks...")
    callbacks = build_callbacks(cfg.callbacks)
    LOGGER.info("Building callbacks...DONE!")

    # build loggers
    LOGGER.info("Building loggers...")
    loggers = build_loggers(cfg.logger)
    LOGGER.info("Building loggers...DONE!")

    # build lightning data module
    LOGGER.info(f"Building datamodule <{cfg.data._target_}>...")
    datamodule = hydra.utils.instantiate(cfg.data)
    assert isinstance(datamodule, pl.LightningDataModule)
    LOGGER.info(f"Building datamodule <{cfg.data._target_}>...DONE!")

    # build model lightning module
    LOGGER.info(f"Building model <{cfg.model._target_}>...")
    model = hydra.utils.instantiate(cfg.model)
    assert isinstance(model, pl.LightningModule)
    LOGGER.info(f"Building model <{cfg.model._target_}>...DONE!")

    # build trainer
    LOGGER.info("Building trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )
    assert isinstance(trainer, pl.Trainer)
    LOGGER.info("Building trainer...DONE!")

    # log hyperparameters
    obj_dict = {
        "cfg": cfg,
        "datmodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "loggers": loggers,
        "trainer": trainer,
    }
    if loggers is not None:
        LOGGER.info("Logging hyperparameters...")
        log_hyperparams(obj_dict)

    # train the model
    if cfg.get("train"):
        LOGGER.info("Training the model...")
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint")
        )
        LOGGER.info("Training the model...DONE!")
    train_metrics = trainer.callback_metrics

    # optionally test the model
    if cfg.get("test"):
        LOGGER.info("Testing the model...")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "" or ckpt_path is None:
            LOGGER.warning("No checkpoint found. Using current weights...")
            ckpt_path = None
        LOGGER.info(f"Best checkpoint saved at: {ckpt_path}.")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        LOGGER.info("Testing the model...DONE!")

    train_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **trainer.callback_metrics}

    return metric_dict, obj_dict


@hydra.main(**HYDRA_CFG)
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    apply_extras(cfg)

    # NOTE: resolve the "received 0 items of ancdata" error
    # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/3
    torch.multiprocessing.set_sharing_strategy("file_system")

    # train the model
    metric_dict, _ = train(cfg)

    # retrieve the metric value for hydra-based hyperaparameter tuning
    metric_values = get_metric_value(metric_dict, cfg.get("optimized_metric"))

    return metric_values


if __name__ == "__main__":
    main()
