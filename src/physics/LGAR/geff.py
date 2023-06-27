"""A file to hold all Geff functions"""
import logging
import torch

from src.physics.LGAR.utils import (
    calc_se_from_theta,
    calc_h_from_se,
    calc_se_from_h,
    calc_k_from_se,
)

log = logging.getLogger("physics.LGAR.Geff")
torch.set_default_dtype(torch.float64)


def calc_geff(use_closed_form_G, soils_data, theta_1, theta_2, nint, device):
    """
  /***********************************************************************************************/
    /* This function calculates the unsaturated capillary drive Geff(0i,0o)  (note "0" are thetas) */
    /* for the Green and Ampt redistribution function.                                             */
    /* used for redistribution following the equation published by Ogden and Saghafian 1995.       */
    /* to compile: "gcc one_block.c -lm"                                                           */
    /*                                                                                             */
    /* author: Fred Ogden, June, 2021,                                                             */
    /***********************************************************************************************/
    :param use_closed_form_G:
    :param soils_data:
    :param theta_1:
    :param theta_2:
    :param nint:
    :return:
    """
    """
    # Theta_1 = theta, theta_2 = soils_data["theta_<x>"] in most cases
    if use_closed_form_G is False:
        # note: units of h in cm.  units of K in cm/s
        # double h2;         // the head at the right-hand side of the trapezoid being integrated [m]
        # double dh;         // the delta h over which integration is performed [m]
        # double Se1,Se2;    // the scaled moisture content on left- and right-hand side of trapezoid
        # double K1,K2;      // the K(h) values on the left and right of the region dh integrated [m]

        # scaled initial water content (0-1) [-]
        se_i = calc_se_from_theta(theta_1, soils_data["theta_e"], soils_data["theta_r"])
        # scaled final water content (0-1) [-]
        se_f = calc_se_from_theta(
            theta_2, soils_data["theta_e"], soils_data["theta_r"]
        )

        # capillary head associated with Se_i [cm]
        h_i = calc_h_from_se(
            se_i, soils_data["alpha"], soils_data["m"], soils_data["n"]
        )
        # capillary head associated with Se_f [cm]
        h_f = calc_h_from_se(
            se_f, soils_data["alpha"], soils_data["m"], soils_data["n"]
        )

        # /* if the lower limit of integration is less than h_min FIXME?? */
        if h_i < soils_data["h_min_cm"]:
            # commenting out as this is not used in the Python version
            return soils_data["h_min_cm"]

        se_inverse_i = calc_se_from_h(h_i, soils_data['alpha'], soils_data['m'], soils_data['n'])
        log.debug(
            f"Se_i = {se_i.item()},  Se_inverse = {se_inverse_i.item()}"
        )
        se_inverse_f = calc_se_from_h(h_f, soils_data['alpha'], soils_data['m'], soils_data['n'])
        log.debug(
            f"Se_f = {se_f.item()},  Se_inverse = {se_inverse_f.item()}"
        )

        # nint = number of "dh" intervals to integrate over using trapezoidal rule
        dh = (h_f - h_i) / nint
        geff = torch.tensor(0.0, device=device)

        # integrate k(h) dh from h_i to h_f, using trapezoidal rule, with subscript
        # 1 denoting the left-hand side of the trapezoid, and 2 denoting the right-hand side
        k1 = calc_k_from_se(se_i, soils_data["ksat_cm_per_h"], soils_data["m"])
        h2 = h_i + dh

        for i in range(nint):
            se2 = calc_se_from_h(
                h2, soils_data["alpha"], soils_data["m"], soils_data["n"]
            )
            k2 = calc_k_from_se(se2, soils_data["ksat_cm_per_h"], soils_data["m"])
            geff = geff + ((k1 + k2) * (dh / 2.0))  # trapezoidal rule
            # reset for next time through loop
            k1 = k2
            h2 = h2 + dh

        geff = torch.abs(
            geff / soils_data["ksat_cm_per_h"]
        )  # by convention Geff is a positive quantity
        log.debug(f"Capillary suction (G) = {geff.item()}")
    else:
        se_f = calc_se_from_theta(
            theta_1, soils_data["theta_e"], soils_data["theta_r"]
        )  # the scaled moisture content of the wetting front
        se_i = calc_se_from_theta(
            theta_2, soils_data["theta_e"], soils_data["theta_r"]
        )  # the scaled moisture content below the wetting front
        h_c = (
            soils_data["bc_psib_cm"]
            * ((2 + 3 * soils_data["bc_lambda"]))
            / (1 + 3 * soils_data["bc_lambda"])
        )  # Green ampt capillary drive parameter, which can be used in the approximation of G with the Brooks-Corey model (See Ogden and Saghafian, 1997)

        geff = h_c * (torch.pow(se_i, (3 + 1 / soils_data["bc_lambda"]))) - torch.pow(
            se_f, (3 + 1 / soils_data["bc_lambda"])
        ) / (1 - torch.pow(se_f, (3 + 1 / soils_data["bc_lambda"])))

        if torch.isinf(geff) or torch.isnan(geff):
            geff = h_c
    return geff
