import logging

from omegaconf import DictConfig
import torch

from dpLGAR.datazoo.basedataset import BaseDataset
from dpLGAR.lgar.mass_balance.global_mass_balance import BaseMassBalance

log = logging.getLogger(__name__)


def get_mass_balance(cfg: DictConfig) -> BaseMassBalance:
    """Creates the global mass balance that we're using within LGAR

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------

    Raises
    ------
    NotImplementedError
        If there is an issue creating the soil layers from the given data
    """

    raise NotImplementedError(f"No Mass Balance Implemented")

