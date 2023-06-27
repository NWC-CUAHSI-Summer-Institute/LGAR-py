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
        self.theta = attributes[global_params.soil_property_indexes["theta_init"]]
        self.dzdt = torch.tensor(0.0, device=global_params.device)
        self.theta_r = attributes[global_params.soil_property_indexes["theta_r"]]
        self.theta_e = attributes[global_params.soil_property_indexes["theta_e"]]
        self.m = attributes[global_params.soil_property_indexes["m"]]
        self.se = calc_se_from_theta(self.theta, self.theta_e, self.theta_r)
        self.psi_cm = global_params.initial_psi
        self.ksat_cm_per_h = calc_k_from_se(self.se, ksat, self.m)
        self.bottom_flag = bottom_flag

    def deepcopy(self, wf):
        """
        Creating a copy of the wf object. The tensors need to be clones to ensure that the objects are not manipulated
        :param wf:
        :return:
        """
        # TODO, we may need to detach the tensors here so the gradient is not tracked?? Making copies is annoying
        wf.depth = self.depth.clone()
        wf.layer_num = self.layer_num
        # wf.attributes = self.attributes
        wf.theta = self.theta.clone()
        wf.theta = self.theta_r
        wf.theta_e = self.theta_e
        wf.m = self.m
        wf.dzdt = self.dzdt.clone()
        wf.se = self.se.clone()
        wf.psi_cm = self.psi_cm.clone()
        wf.ksat_cm_per_h = self.ksat_cm_per_h.clone()
        wf.bottom_flag = self.bottom_flag
        return wf

    def is_equal(self, front):
        depth_equal = (front.depth == self.depth)
        psi_cm_equal = (front.psi_cm == self.psi_cm)
        dzdt = (front.dzdt == self.dzdt)
        if depth_equal:
            if psi_cm_equal:
                if dzdt:
                    return True
        return False



