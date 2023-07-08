"""A file to hold all soil functions"""
import logging
import pandas as pd
import torch
from torch import Tensor

log = logging.getLogger("physics.utils")


def calc_theta_from_h(
    h: Tensor, alpha: Tensor, m: Tensor, n: Tensor, theta_e: Tensor, theta_r: Tensor
) -> torch.Tensor():
    """
    function to calculate theta from h

    :parameter h: the initial psi (cm)
    :parameter soil_properties: All soils and their properties
    :parameter device: the device that we're using
    :return thickness of individual layers
    """
    return (
        1.0 / (torch.pow(1.0 + torch.pow(alpha * h, n), m)) * (theta_e - theta_r)
    ) + theta_r


def calc_bc_lambda(m: Tensor) -> Tensor:
    """
    Given van Genuchten parameters calculate estimates of
    Brooks & Corey bc_lambda and bc_psib
    :param alpha: Van Genuchten parameter
    :param n: Van Genuchten parameter
    :return:
    """
    p = 1.0 + (2.0 / m)
    bc_lambda = 2.0 / (p - 3.0)
    return bc_lambda


def calc_m(n):
    m = 1.0 - (1.0 / n)
    return m


def calc_h_min_cm(bc_lambda, bc_psib_cm) -> pd.DataFrame:
    """
    # /* this is the effective capillary drive after */
    # /* Morel-Seytoux et al. (1996) eqn. 13 or 15 */
    # /* psi should not be less than this value.  */
    :param df:
    :return:
    """
    h_min_cm = bc_psib_cm * (2.0 + 3.0 / bc_lambda) / (1.0 + 3.0 / bc_lambda)
    return h_min_cm


def calc_bc_psib(alpha: Tensor, m: Tensor) -> Tensor:
    """
    Given van Genuchten parameters calculate estimates of
    Brooks & Corey bc_psib
    :param alpha:
    :param n:
    :return:
    """
    # m_ = 1.0 - (1.0 / n)
    p_ = 1.0 + (2.0 / m)
    bc_psib = (
        (p_ + 3.0)
        * (147.8 + 8.1 * p_ + 0.092 * p_ * p_)
        / (2.0 * alpha * p_ * (p_ - 1.0) * (55.6 + 7.4 * p_ + p_ * p_))
    )
    return bc_psib


def calc_se_from_theta(theta: Tensor, theta_e: Tensor, theta_r: Tensor) -> Tensor:
    """
    function to calculate Se from theta
    :param theta_init: the calculated inital theta
    :param theta_e: theta_e
    :param theta_r: theta_r
    :return: Se this is the relative (scaled 0-1) water content, like Theta
    """
    return (theta - theta_r)/(theta_e - theta_r)


def calc_se_from_h(h, alpha, m, n):
    """
    function to calculate Se from h
    :param h:
    :param alpha:
    :param m:
    :param n:
    :return:
    """
    h_abs = torch.abs(h)
    if h_abs < 1.0e-01:
        return torch.tensor(
            1.0
        )  # TODO EXPLORE A CLAMP (this function doesn't work well for tiny h)
    return 1.0 / (torch.pow(1.0 + torch.pow(alpha * h, n), m))


def calc_k_from_se(se: Tensor, ksat: Tensor, m: Tensor) -> Tensor:
    """
    function to calculate K from Se
    :param se: this is the relative (scaled 0-1) water content, like Theta
    :param ksat: saturated hydraulic conductivity
    :param m: Van Genuchten
    :return: hydraulic conductivity (K)
    """
    return (
        ksat
        * torch.sqrt(se)
        * torch.pow(1.0 - torch.pow(1.0 - torch.pow(se, 1.0 / m), m), 2.0)
    )


def calc_h_from_se(
    se: torch.Tensor, alpha: torch.Tensor, m: torch.Tensor, n: torch.Tensor
):
    """
    function to calculate h from Se
    """
    return 1.0 / alpha * torch.pow(torch.pow(se, (-1.0 / m)) - 1.0, (1.0 / n))