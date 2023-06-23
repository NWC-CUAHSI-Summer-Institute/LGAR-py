"""A file to hold all soil functions"""
import logging
import torch

from src.physics.LGAR.utils import read_soils
from src.physics.LGAR.geff import calc_geff

log = logging.getLogger("physics.LGAR.dry_depth")
torch.set_default_dtype(torch.float64)


def calc_dry_depth(lgar, use_closed_form_G, nint, timestep_h):
    """
     /* This routine calculates the "dry depth" of a newly created wetting front in the top soil layer after
    a non-rainy period or a big increase in rainrate  on an unsaturated first layer.
    Note: Calculation of the initial depth of a new wetting front in the first layer uses the concept of "dry depth",
    described in the 2015 GARTO paper (Lai et al., An efficient and guaranteed stable numerical method ffor
    continuous modeling of infiltration and redistribution with a shallow dynamic water table). */
    :param lgar: the LGAR model obj
    :param use_closed_form_G:
    :param nint:
    :param timestep_h:
    :param delta_theta:
    :return:
    """
    head_index = 0  # 0 is always the starting index
    current_front = lgar.wetting_fronts[0]

    #  these are the limits of integration
    # Theta_1 = current.theta, theta_2 = theta_e
    soils_data = read_soils(lgar, current_front)
    delta_theta = (
        soils_data["theta_e"] - current_front.theta
    )  # water content of the first (most surficial) existing wetting front
    tau = timestep_h * soils_data["ksat_cm_per_h"] / delta_theta

    geff = calc_geff(
        use_closed_form_G,
        soils_data,
        current_front.theta,
        soils_data["theta_e"],
        nint,
        lgar.device,
    )

    # note that dry depth originally has a factor of 0.5 in front
    dry_depth = 0.5 * (tau + torch.sqrt(tau * tau + 4.0 * tau * geff))

    # when dry depth greater than layer 1 thickness, set dry depth to layer 1 thickness
    dry_depth = torch.min(lgar.cum_layer_thickness[current_front.layer_num], dry_depth)

    return dry_depth
