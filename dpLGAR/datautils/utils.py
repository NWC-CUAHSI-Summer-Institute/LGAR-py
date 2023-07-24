import logging

from omegaconf import DictConfig
import torch
from torch import Tensor

from dpLGAR.lgar.utils import (
    calc_theta_from_h,
    calc_bc_lambda,
    calc_bc_psib,
    calc_h_min_cm,
    calc_m,
)

log = logging.getLogger(__name__)


def generate_soil_metrics(
    cfg: DictConfig,
    alpha: Tensor,
    n: Tensor,
    theta_e: Tensor,
    theta_r: Tensor,
) -> Tensor:
    """
    Reading the soils dataframe
    :param cfg: the config file
    :param wilting_point_psi_cm wilting point (the amount of water not available for plants or not accessible by plants)

    Below are the variables used inside of the soils dataframe:
    Texture: The soil classification
    theta_r: Residual Water Content
    theta_e: Wilting Point
    alpha(cm^-1): ???"
    n: ???
    m: ???
    Ks(cm/h): Saturated Hydraulic Conductivity

    :return:
    """
    h = torch.tensor(
        cfg.data.wilting_point_psi_cm, device=cfg.device
    )  # Wilting point in cm
    initial_psi = torch.tensor(cfg.data.initial_psi, device=cfg.device)
    m = calc_m(n)
    theta_wp = calc_theta_from_h(h, alpha, m, n, theta_e, theta_r)
    theta_init = calc_theta_from_h(
        initial_psi, alpha.clone(), m, n, theta_e, theta_r
    )
    bc_lambda = calc_bc_lambda(m)
    bc_psib_cm = calc_bc_psib(alpha, m)
    h_min_cm = calc_h_min_cm(bc_lambda, bc_psib_cm)

    soils_data = torch.stack(
        [
            theta_r,
            theta_e,
            theta_wp,
            theta_init,
            m,
            bc_lambda,
            bc_psib_cm,
            h_min_cm,
            alpha,
            n,
        ]
    )  # Putting all numeric columns in a tensor other than the Texture column
    return soils_data.transpose(0, 1)
