"""A file to hold all soil functions"""
import logging
import torch

from src.physics.LGAR.utils import read_soils
from src.physics.LGAR.geff import calc_geff
from src.physics.LGAR.utils import calc_theta_from_h, calc_se_from_theta, calc_k_from_se

log = logging.getLogger("physics.LGAR.dzdt")
torch.set_default_dtype(torch.float64)


def calc_dzdt(lgar, use_closed_form_G, nint, h_p):
    """
    code to calculate velocity of fronts
    equations with full description are provided in the lgar paper (currently under review) */
    :param use_closed_form_G:
    :param nint:
    :param ponded_depth_subtimestep_cm:
    :return:
    """
    log.debug(f"Calculating dz/dt")

    dzdt = torch.tensor(0.0, device=lgar.device)
    for i in range(len(lgar.wetting_fronts) - 1):
        """
        we're done calculating dZ/dt's when at the end of the list (hence len -1)
        """
        current = lgar.wetting_fronts[i]
        layer_num = current.layer_num  # what layer the front is in
        k_cm_per_h = current.k_cm_per_h  # K(theta)

        if k_cm_per_h <= 0:
            log.debug(f"K is zero: layer:{layer_num} {k_cm_per_h}")
            raise ValueError("K is zero")

        depth_cm = current.depth_cm
        soils_data = read_soils(lgar, current)

        next_ = lgar.wetting_fronts[i + 1]
        theta_1 = next_.theta
        theta_2 = current.theta

        # needed for multi-layered dz/dt equation.  Equal to sum from n=1 to N-1 of (L_n/K_n(theta_n))
        bottom_sum = torch.tensor(0.0, device=lgar.device)

        if current.to_bottom:
            current.dzdt_cm_per_h = torch.tensor(0.0, device=lgar.device)
        else:  # Since this is a replacement for a while loop, we have to use this else loop to replace a continue
            if layer_num > 0:  # Theory is this is supposed to be making sure that we're not using the bottom "0-base lists"
                bottom_sum = (
                    bottom_sum
                    + (depth_cm - lgar.cum_layer_thickness[layer_num - 1]) / k_cm_per_h
                )
            else:
                if theta_1 > theta_2:
                    log.error("theta_1 cannot be larger than theta_2")
                    raise ValueError

            geff = calc_geff(use_closed_form_G, soils_data, theta_1, theta_2, nint, lgar.device)
            delta_theta = current.theta - next_.theta

            if current.layer_num == 0:  # this front is in the upper layer
                if delta_theta > 0:
                    dzdt = (
                        1.0
                        / delta_theta
                        * (
                            soils_data["ksat_cm_per_h"] * (geff + h_p) / current.depth_cm
                            + current.k_cm_per_h
                        )
                    )
                else:
                    dzdt = torch.tensor(0.0, device=lgar.device)

            else:  # we are in the second or greater layer
                denominator = bottom_sum

                for k in range(1, layer_num):
                    soil_num_loc = lgar.layer_soil_type[
                        layer_num - k
                    ]  # _loc denotes the soil_num is local to this loop
                    theta_prev_loc = calc_theta_from_h(
                        current.psi_cm, soils_data, device=lgar.device
                    )

                    se_prev_loc = calc_se_from_theta(
                        theta_prev_loc, soils_data["theta_e"], soils_data["theta_r"]
                    )

                    k_cm_per_h_prev_loc = calc_k_from_se(
                        se_prev_loc, soils_data["ksat_cm_per_h"], soils_data["m"]
                    )

                    denominator = denominator + (
                        (lgar.cum_layer_thickness[k] - lgar.cum_layer_thickness[k - 1])
                        / k_cm_per_h_prev_loc
                    )

                numerator = depth_cm  # + (Geff +h_p)* Ksat_cm_per_h / K_cm_per_h;

                if delta_theta > 0:
                    dzdt = (1.0 / delta_theta) * (
                        (numerator / denominator)
                        + soils_data["ksat_cm_per_h"] * (geff + h_p) / depth_cm
                    )
                else:
                    dzdt = torch.tensor(0.0, device=lgar.device)

            current.dzdt_cm_per_h = dzdt
