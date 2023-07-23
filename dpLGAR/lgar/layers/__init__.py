import logging

from omegaconf import DictConfig
import torch

from dpLGAR.datazoo.basedataset import BaseDataset
from dpLGAR.lgar.layers.Layer import Layer

log = logging.getLogger(__name__)


def create_layers(cfg: DictConfig, ds: BaseDataset, alpha, n, ksat, theta_e, theta_r) -> Layer:
    """Creates the soil profile that we're examining within LGAR

    Parameters
    ----------
    cfg : Config
        The run configuration.
    ds: BaseDataset
        The containing the soil attributes
    alpha: Torch.nn.Parameter
        The Van Genuhten parameter representing the inverse of air-entry
    n: Torch.nn.Parameter
        The Van Genuhten shape parameters
    ksat: Torch.nn.Parameter
        Effective saturated hydraulic conductivity
    theta_e:
        Water content at effective saturation
    theta_r:
        Residual water content
    Returns
    -------
        A soil layer object
    Raises
    ------
    NotImplementedError
        If there is an issue creating the soil layers from the given data
    """

    top_layer = Layer()

    # raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

