"""A file to hold all wetting_front utils functions"""
import logging
import torch

from src.physics.LGAR.utils import (
    calc_se_from_theta,
    calc_h_from_se,
    calc_se_from_h,
    calc_k_from_se,
)

from src.physics.wetting_fronts.WettingFront import WettingFront
from src.physics.LGAR.utils import read_soils

log = logging.getLogger("physics.wetting_fronts.utils")
torch.set_default_dtype(torch.float64)


def create_surficial_front_func(lgar, ponded_depth_cm, volin, dry_depth):
    """
    // ######################################################################################
    /* This subroutine is called iff there is no surfacial front, it creates a new front and
       inserts ponded depth, and will return some amount if can't fit all water */
    // ######################################################################################
    """

    to_bottom = False
    top_front = lgar.wetting_fronts[0]  # Specifically pointing to the first front
    soils_data = read_soils(lgar, top_front)

    layer_num = 0

    delta_theta = soils_data["theta_e"] - top_front.theta

    if (
            dry_depth * delta_theta
    ) > ponded_depth_cm:  # all the ponded depth enters the soil
        volin = ponded_depth_cm
        theta_new = torch.min(
            (top_front.theta + ponded_depth_cm / dry_depth), soils_data["theta_e"]
        )
        surficial_front = WettingFront(dry_depth, theta_new, layer_num, to_bottom)
        lgar.wetting_fronts.insert(
            0, surficial_front
        )  # inserting the new front in the front layer
        ponded_depth_cm = torch.tensor(0.0, device=lgar.device)
    else:  # // not all ponded depth fits in
        volin = dry_depth * delta_theta
        ponded_depth_cm = lgar.ponded_depth_cm - (dry_depth * delta_theta)
        theta_new = soils_data["theta_e"]  # fmin(theta1 + (*ponded_depth_cm) /dry_depth, theta_e)
        if (
                dry_depth < lgar.cum_layer_thickness[0]
        ):  # checking against the first layer
            surficial_front = WettingFront(
                dry_depth, soils_data["theta_e"], layer_num, to_bottom
            )
        else:
            surficial_front = WettingFront(
                dry_depth, soils_data["theta_e"], layer_num, True
            )
        lgar.wetting_fronts.insert(0, surficial_front)

    # These calculations are allowed as we're creating a dry layer of the same soil type
    se = calc_se_from_theta(theta_new, soils_data["theta_e"], soils_data["theta_r"])
    new_front = lgar.wetting_fronts[0]
    new_front.psi_cm = calc_h_from_se(
        se, soils_data["alpha"], soils_data["m"], soils_data["n"]
    )

    new_front.k_cm_per_h = (
            calc_k_from_se(se, soils_data["ksat_cm_per_h"], soils_data["m"])
            * lgar.frozen_factor[layer_num]
    )  # // AJ - K_temp in python version for 1st layer
    new_front.dzdt_cm_per_h = 0.0  # for now assign 0 to dzdt as it will be computed/updated in lgar_dzdt_calc function

    return ponded_depth_cm, volin