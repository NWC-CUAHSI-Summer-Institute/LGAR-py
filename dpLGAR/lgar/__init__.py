import logging

from omegaconf import DictConfig
import torch


log = logging.getLogger(__name__)


def initialize_physics(cfg: DictConfig, ds: BaseDataset, ponded_depth) -> None:
    """Creates the soil profile that we're examining within LGAR

    Parameters
    ----------
    cfg : Config
        The run configuration.
    ds: BaseDataset
        The containing the soil attributes
    ponded_depth_max: Torch.nn.Parameter
        The maximum ponded depth in the soils
    Returns
    -------
        A soil layer object
    Raises
    ------
    NotImplementedError
        If there is an issue creating the soil layers from the given data
    """

    # raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

