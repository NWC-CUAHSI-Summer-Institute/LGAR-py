from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

from lgartorch.models.physics.utils import (
    calc_theta_from_h,
    calc_se_from_theta,
    calc_h_from_se,
)

log = logging.getLogger("models.physics.lgar.green_ampt")


def calc_geff(global_params, pet, psi_cm, theta_e, theta_r, m, alpha, n) -> Tensor:
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
    :param global_params:
    :param pet:
    :param psi_cm:
    :param theta_e:
    :param theta_r:
    :param m:
    :param alpha:
    :param n:
    :return:
    """
    pass