from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.models.physics.layers.WettingFront import WettingFront
from lgartorch.models.physics.lgar.aet import calculate_aet

log = logging.getLogger("models.physics.layers.Layer")


class Layer:
    def __init__(
        self,
        global_params,
        layer_index: int,
        c: Tensor,
        alpha: torch.nn.Parameter,
        n: torch.nn.Parameter,
        ksat: torch.nn.Parameter,
        texture_map: dict,
    ):
        """
        A layer of soil (within the soil stack).
        Each soil layer can have many wetting fronts and several properties
        :param cfg: The DictConfig
        :param global_params: many of the values within the cfg file, but as tensors
        :param c: All soil attributes
        :param alpha: All alpha van Genuchten params
        :param n: All n van Genuchten params
        :param ksat: All saturated hydraulic conductivity params
        :param is_top: TBD if this is necessary. Rn it's always true
        """
        super().__init__()
        self.global_params = global_params
        self.layer_num = layer_index
        self.layer_thickness = self.global_params.layer_thickness_cm[self.layer_num]
        self.cumulative_layer_thickness = self.global_params.cum_layer_thickness[
            self.layer_num
        ]
        self.soil_type = self.global_params.layer_soil_type[self.layer_num]
        self.texture = texture_map[self.soil_type]
        self.attributes = c[self.soil_type]
        self.alpha_layer = alpha[self.soil_type]
        self.n_layer = n[self.soil_type]
        self.ksat_layer = ksat[self.soil_type]
        self.wetting_fronts = []
        self.wetting_fronts.append(
            WettingFront(
                self.global_params,
                self.cumulative_layer_thickness,
                self.attributes,
                self.ksat_layer,
            )
        )
        self.next_layer = None
        if (
            layer_index < global_params.num_layers - 1
        ):  # Checking to see if there is a layer below this one
            self.next_layer = Layer(
                global_params, layer_index + 1, c, alpha, n, ksat, texture_map
            )

    def input_precip(self, precip: Tensor) -> None:
        raise NotImplementedError

    def calc_aet(self, pet: Tensor) -> None:
        """
        ONLY CALLED FROM TOP LAYER
        Calculates the Actual Evapotranspiration for each layer
        :param pet: Potential evapotranspiration
        :param subcycle_length_h: the length of each subcycle step (in hours)
        :return:
        """
        top_wetting_front = self.wetting_fronts[0]
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_property_indexes["theta_r"]]
        m = self.attributes[self.global_params.soil_property_indexes["m"]]
        aet = calculate_aet(
            self.global_params,
            pet,
            top_wetting_front.psi_cm,
            theta_e,
            theta_r,
            m,
            self.alpha_layer,
            self.n_layer,
        )
        return aet

    def is_saturated(self):
        """
        Determining if the top layer's first wetting front is saturated
        If saturated, then there will be runoff
        :return:
        """
        top_wetting_front = self.wetting_fronts[0]
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        return True if top_wetting_front.theta >= theta_e else False

    def mass_balance(self) -> Tensor:
        """
        A function that calculates the mass inside of the current layer
        If `next_layer` is not None, then we iterate through the soil stack to
        find the mass underneath
        :return:
        """
        sum = torch.tensor(0, dtype=torch.float64)
        if self.layer_num == 0:
            base_depth = torch.tensor(0.0, device=self.global_params.device)
        else:
            # The base depth is the depth at the top of the layer
            base_depth = self.cumulative_layer_thickness - self.layer_thickness
        if len(self.wetting_fronts) > 1:
            # TODO TEST THIS!!!
            # Iterate through the list elements except the last one
            for i, wf in enumerate(self.wetting_fronts[:-1]):
                current_front = self.wetting_fronts[i]
                next_front = self.wetting_fronts[i + 1]
                sum = sum + (current_front.depth - base_depth) * (
                    current_front.theta - next_front.theta
                )
            last_front = self.wetting_fronts[-1]
            sum = sum + (last_front.depth - base_depth) * last_front.theta
        else:
            current_front = self.wetting_fronts[0]
            sum = sum + (current_front.depth - base_depth) * current_front.theta
        if self.next_layer is not None:
            return sum + self.next_layer.mass_balance()
        else:
            return sum
