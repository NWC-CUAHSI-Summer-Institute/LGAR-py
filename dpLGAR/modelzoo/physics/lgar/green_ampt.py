from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

from dpLGAR.modelzoo.physics.utils import (
    calc_theta_from_h,
    calc_k_from_se,
    calc_se_from_h,
    calc_se_from_theta,
    calc_h_from_se,
    safe_pow
)

log = logging.getLogger("modelzoo.physics.lgar.green_ampt")


def calc_geff(global_params, attributes, theta_1, theta_2, alpha, n, ksat) -> Tensor:
    """
    /***********************************************************************************************/
    /* This function calculates the unsaturated capillary drive Geff(0i,0o)  (note "0" are thetas) */
    /* for the Green and Ampt redistribution function.                                             */
    /* used for redistribution following the equation published by Ogden and Saghafian 1995.       */
    /* to compile: "gcc one_block.c -lm"                                                           */
    /*                                                                                             */
    /* author: Fred Ogden, June, 2021,                                                             */
    /***********************************************************************************************/

    :param global_params:
    :param attributes:
    :param theta_1:
    :param theta_2:
    :param alpha_layer:
    :param n_layer:
    :param ksat_layer:
    :return:
    """
    theta_r = attributes[global_params.soil_index["theta_r"]]
    theta_e = attributes[global_params.soil_index["theta_e"]]
    m = attributes[global_params.soil_index["m"]]
    bc_lambda = attributes[global_params.soil_index["bc_lambda"]]
    bc_psib_cm = attributes[global_params.soil_index["bc_psib_cm"]]
    h_min_cm = attributes[global_params.soil_index["h_min_cm"]]
    if global_params.use_closed_form_G is False:
        # note: units of h in cm.  units of K in cm/s
        # double h2;         // the head at the right-hand side of the trapezoid being integrated [m]
        # double dh;         // the delta h over which integration is performed [m]
        # double Se1,Se2;    // the scaled moisture content on left- and right-hand side of trapezoid
        # double K1,K2;      // the K(h) values on the left and right of the region dh integrated [m]

        # scaled initial water content (0-1) [-]
        se_i = calc_se_from_theta(theta_1, theta_e, theta_r)
        # scaled final water content (0-1) [-]
        se_f = calc_se_from_theta(theta_2, theta_e, theta_r)

        # capillary head associated with Se_i [cm]
        h_i = calc_h_from_se(se_i, alpha, m, n)
        # capillary head associated with Se_f [cm]
        h_f = calc_h_from_se(se_f, alpha, m, n)

        # Checkpoint
        se_inverse_i = calc_se_from_h(h_i, alpha, m, n)
        se_inverse_f = calc_se_from_h(h_f, alpha, m, n)

        # nint = number of "dh" intervals to integrate over using trapezoidal rule
        dh = (h_f - h_i) / global_params.nint
        geff = torch.tensor(0.0, device=global_params.device)

        # integrate k(h) dh from h_i to h_f, using trapezoidal rule, with subscript
        # 1 denoting the left-hand side of the trapezoid, and 2 denoting the right-hand side
        k1 = calc_k_from_se(se_i, ksat, m)
        h2 = h_i + dh

        for i in range(global_params.nint):
            # trapezoidal rule
            se2 = calc_se_from_h(h2, alpha, m, n)
            k2 = calc_k_from_se(se2, ksat, m)
            geff = geff + ((k1 + k2) * (dh / 2.0))
            k1 = k2
            h2 = h2 + dh

        # by convention Geff is a positive quantity
        geff = torch.abs(geff / ksat)
    else:
        # the scaled moisture content of the wetting front
        se_f = calc_se_from_theta(theta_1, theta_e, theta_r)
        # the scaled moisture content below the wetting front
        se_i = calc_se_from_theta(theta_2, theta_e, theta_r)
        # Green ampt capillary drive parameter, which can be used in the approximation of G with the Brooks-Corey model (See Ogden and Saghafian, 1997)
        h_c = bc_psib_cm * (2 + 3 * bc_lambda) / (1 + 3 * bc_lambda)

        geff = h_c * (torch.pow(se_i, (3 + 1 / bc_lambda))) - torch.pow(
            se_f, (3 + 1 / bc_lambda)
        ) / (1 - torch.pow(se_f, (3 + 1 / bc_lambda)))

        if torch.isinf(geff) or torch.isnan(geff):
            geff = h_c
    return geff
