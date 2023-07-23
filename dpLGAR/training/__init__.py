import logging
import warnings
from typing import List

from omegaconf import DictConfig
import torch

# import neuralhydrology.training.loss as loss

log = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.

    Currently only 'Adam' is supported.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be optimized.
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training.
    """
    if cfg.modelzoo.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.modelzoo.learning_rate)
    else:
        raise NotImplementedError(f"{cfg.optimizer} not implemented or not linked in `get_optimizer()`")

    return optimizer


def get_loss_obj(cfg: DictConfig):
    """Get loss object, depending on the run configuration.

    Currently supported are 'MSE', 'NSE', 'RMSE', 'GMMLoss', 'CMALLoss', and 'UMALLoss'.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the
        head.
    """
    # TODO ADD Range bound Loss custom
    if cfg.loss.lower() == "mse":
        loss_obj = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss_obj
