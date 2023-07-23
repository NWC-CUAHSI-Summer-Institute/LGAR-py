from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger("modelzoo.physics.lgar.frozen_factor")


def frozen_factor_hydraulic_conductivity() -> None:
    """
    calculates frozen factor based on L. Wang et al. (www.hydrol-earth-syst-sci.net/14/557/2010/)
    uses layered-average soil temperatures and an exponential function to compute frozen fraction
    for each layer
    :return: None
    """
    raise NotImplementedError