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
"""Helper functions for implementing `Rich` library."""
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax as rsyntax
import rich.tree as rtree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from seneva.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if they are not given.

    Args:
        cfg (DictConfig): The configuration dictionary to be checked.
        save_to_file (bool, optional): Whether to save the tags to a file in
            Hydra output folder. Defaults to False.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        LOGGER.info("No tags specified. Now prompting user to input tags.")
        tags = Prompt.ask("Please input tags (separated by comma):")
        tags = [tag.strip() for tag in tags.split(",") if tag != ""]
        with open_dict(cfg):
            cfg.tags = tags
        LOGGER.info(f"Tags updated: {tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.txt"), "w") as f:
            rich.print(cfg.tags, file=f)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = [
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ],
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print the configuration tree with `Rich` library.

    Args:
        cfg (DictConfig): The configuration dictionary to be printed.
        print_order (Sequence[str], optional): The order of the components to
            be printed. Defaults to ["data", "model", "callbacks", "logger",
            "trainer", "paths", "extras"].
        resolve (bool, optional): Whether to resolve the configuration.
            Defaults to False.
        save_to_file (bool, optional): Whether to save the configuration to a
            file in Hydra output folder. Defaults to False.
    """
    style = "dim"
    tree = rtree.Tree("CONFIG", style=style, guide_style=style)
    queue = []

    # add fields to queue following the print order
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            LOGGER.info(f"Field '{field}' not found in config, skipping...")

    # add the rest other fields to the queue
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate configuration tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_item = cfg[field]
        if isinstance(config_item, DictConfig):
            branch_content = OmegaConf.to_yaml(config_item, resolve=resolve)
        else:
            branch_content = str(config_item)

        branch.add(rsyntax.Syntax(branch_content, "yaml", line_numbers=False))

    # print the configuration tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.txt"), "w") as f:
            rich.print(tree, file=f)
