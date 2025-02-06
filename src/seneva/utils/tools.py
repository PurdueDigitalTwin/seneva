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
"""Helper functions library."""
import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from omegaconf import DictConfig

from seneva.utils.logging import get_logger
from seneva.utils.rich_utils import enforce_tags, print_config_tree

# Constants
LOGGER = get_logger(__name__)


def apply_extras(cfg: DictConfig) -> None:
    """Apply extra settings ahead of initializations.

    .. note::
        This function is called before Hydra initialization to apply extra
        system level settings:
            - `ignore_warnings`: Ignoring python warnings
            - `enfoce_tags`: Enforcing tags for the experiment
            - `print_config`: Printing the config tree with `Rich` library

    Args:
        cfg (DictConfig): The configuration dictionary.
    """
    # skip if no `extras` section is found
    if not cfg.get("extras"):
        LOGGER.info("Extras section not found. Skipping...")
        return

    extras: DictConfig = cfg.extras
    # disable python warnings
    if extras.get("ignore_warnings"):
        LOGGER.info("Disabling Python warnings! <extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # enforce tags by prompting users to input tags from command line
    if extras.get("enforce_tags"):
        LOGGER.info("Enforcing tags! <extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print the config tree using ``Rich`` library
    if extras.get("print_config"):
        LOGGER.info("Printing config tree! <extras.print_config=True>")
        print_config_tree(cfg)


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str] = None
) -> Optional[float]:
    """Safely retrieves the metric values logged by LightningModule.

    Args:
        metric_dict (Dict[str, Any]): The dictionary containing the metric
            values logged by LightningModule.
        metric_name (Optional[str], optional): The name of the metric to be
            retrieved. Defaults to None.

    Returns:
        Optional[float]: The metric value if found, otherwise `None`.
    """
    if metric_name is None:
        LOGGER.info("No metric name specified. Skipping...")
        return None

    if metric_name not in metric_dict:
        raise KeyError(
            f"Metric <metric_name={metric_name}> not found!\n"
            "Make sure the metric is correctly logged in LightningModule!\n"
            "Make sure `optimized_metric` name in hyperparameter search"
            "configs are correct!"
        )

    metric_value = metric_dict[metric_name].item()
    LOGGER.debug(f"Retrieved metric value <{metric_name}={metric_value}>")

    return metric_value


def task_wrapper(
    fn: Callable[
        [
            DictConfig,
        ],
        tuple,
    ]
) -> Callable:
    """A wrapper for task functions handling failure behavior.

    .. note::
        This function is used as a decorator for task functions. It handles
        the failure behavior of the task function, and can be extended for
        other purposes:
            - Ensure closing the loggers for multirun scenarios.
            - Log and save the exception messages to a `.log` file.
            - Mark the failed run with a dedicated file in the `logs/` folder.
            - other purposes...

    Args:
        fn (Callable[[DictConfig,], tuple]): The task function to be wrapped.
        The task function should take a configuration dictionary as input and
        return a tuple of metric dictionary and object dictionary.
    """

    def wrapped(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Wrapped task function.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: The metric dictionary and
                the object dictionary returned by the task function.
        """
        # try execute the task function
        try:
            metric_dict, object_dict = fn(cfg)
        # catch any exception
        except Exception as e:
            # save the exception message to a `.log` file
            LOGGER.exception(e)

            # NOTE: some hyperparameter combination in searching procedure
            # may be invalid or cause out-of-memory error. In these cases,
            # you might want to comment out the following line.
            raise e
        finally:
            # display the output directory
            LOGGER.info(f"All logs are saved to {cfg.paths.output_dir}.")

            # close wandb logger if it is used
            if find_spec("wandb"):
                import wandb  # type: ignore

                if wandb.run:
                    LOGGER.info("Closing wandb logger...")
                    wandb.finish()
                    LOGGER.info("Closed wandb logger...DONE!")

        return metric_dict, object_dict

    return wrapped


# Register custom resolver for `OmegaConf`
def join_string_underscore(texts: Iterable[str]) -> str:
    """Join strings with underscore."""
    return "_".join(str(t) for t in texts)


def parse_git_sha(clean: bool = False) -> str:
    """Parse git information and export to environment variables.

    Args:
        clean (bool, optional): If to check whether the repository is clean.
            Defaults to `False`.

    Returns:
        str: The SHA in short format of the current git repository.

    Raises:
        RuntimeError: If the repository is not clean when `clean=True`.
    """
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=7)
    is_clean = not repo.is_dirty()

    if clean and not is_clean:
        raise RuntimeError(
            "Clean your git repository before running experiments! "
            f"GIT SHA: {sha}, Status: {'Clean' if is_clean else 'Dirty'}"
        )

    os.environ["PROJECT_GIT_SHA"] = sha
    os.environ["PROJECT_GIT_SHORT_SHA"] = short_sha

    return short_sha
