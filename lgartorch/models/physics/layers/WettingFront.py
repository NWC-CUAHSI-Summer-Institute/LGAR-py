from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.models.physics.utils import calc_se_from_theta, calc_k_from_se

log = logging.getLogger("models.physics.layers.WettingFront")


class WettingFront:
    def __init__(
        self,
        global_params,
        cum_layer_thickness: Tensor,
        attributes: Tensor,
        ksat: torch.nn.Parameter,
        bottom_flag=True,
    ):
        """
        A class that defines the wetting front within the soil layers
        The wetting front will keep track of all mass entering and
        exiting the soil layers. There can be many WettingFronts in a layer
        :param cum_layer_thickness:
        :param attributes:
        :param ksat:
        :param bottom_flag:
        """
        super().__init__()
        self.depth = cum_layer_thickness
        self.theta = attributes[global_params.soil_property_indexes["theta_init"]]
        self.dzdt_cm_per_h = torch.tensor(0.0, device=global_params.device)
        theta_r = attributes[global_params.soil_property_indexes["theta_r"]]
        theta_e = attributes[global_params.soil_property_indexes["theta_e"]]
        m = attributes[global_params.soil_property_indexes["m"]]
        self.se = calc_se_from_theta(self.theta, theta_e, theta_r)
        self.psi_cm = global_params.initial_psi
        self.ksat_cm_per_h = calc_k_from_se(self.se, ksat, m)
        self.bottom_flag = bottom_flag
