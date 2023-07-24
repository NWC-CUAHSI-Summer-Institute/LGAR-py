from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from dpLGAR.modelzoo.physics.utils import (
    calc_se_from_theta,
    calc_k_from_se,
    calc_theta_from_h,
    calc_h_from_se,
)

log = logging.getLogger("modelzoo.physics.layers.WettingFront")


class WettingFront:
    def __init__(
        self,
        soil_state,
        cum_layer_thickness: Tensor,
        layer_num: int,
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
        self.layer_num = layer_num
        # self.attributes = attributes
        self.theta = attributes[soil_state.soil_index["theta_init"]]
        self.dzdt = torch.tensor(0.0)
        theta_r = attributes[soil_state.soil_index["theta_r"]]
        theta_e = attributes[soil_state.soil_index["theta_e"]]
        m = attributes[soil_state.soil_index["m"]]
        self.se = calc_se_from_theta(self.theta, theta_e, theta_r)
        self.psi_cm = soil_state.initial_psi
        self.k_cm_per_h = calc_k_from_se(self.se, ksat, m)
        self.to_bottom = bottom_flag

    def update_soil_parameters(self, global_params, attributes, alpha, n, ksat):
        theta_r = attributes[global_params.soil_index["theta_r"]]
        theta_e = attributes[global_params.soil_index["theta_e"]]
        m = attributes[global_params.soil_index["m"]]
        self.se = calc_se_from_theta(self.theta, theta_e, theta_r)
        self.psi_cm = calc_h_from_se(self.se, alpha, m, n)
        self.k_cm_per_h = calc_k_from_se(self.se, ksat, m)

    def deepcopy(self, wf):
        """
        Creating a copy of the wf object. The tensors need to be clones to ensure that the objects are not manipulated
        :param wf:
        :return:
        """
        # TODO, we may need to detach the tensors here so the gradient is not tracked?? Making copies is annoying
        wf.depth = self.depth.clone()
        wf.layer_num = self.layer_num
        wf.theta = self.theta.clone()
        wf.dzdt = self.dzdt.clone()
        wf.se = self.se.clone()
        wf.psi_cm = self.psi_cm.clone()
        wf.k_cm_per_h = self.k_cm_per_h.clone()
        wf.to_bottom = self.to_bottom
        return wf

    def is_equal(self, front):
        depth_equal = front.depth == self.depth
        psi_cm_equal = front.psi_cm == self.psi_cm
        dzdt = front.dzdt == self.dzdt
        if depth_equal:
            if psi_cm_equal:
                if dzdt:
                    return True
        return False

    def print(self):
        log.info(
            f"[{self.depth.item():.4f}, {self.theta.item():.10f}, {self.layer_num}, {self.dzdt.item():.6f}, {self.k_cm_per_h.item():.6f}, {self.psi_cm:.4f}]"
        )
