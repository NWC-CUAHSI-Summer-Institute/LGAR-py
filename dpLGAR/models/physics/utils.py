"""A file to hold all soil functions"""
import logging
import pandas as pd
import torch
from torch import Tensor

log = logging.getLogger("physics.utils")
zero = torch.tensor(0.0, dtype=torch.float64)
threshold = torch.tensor(1e-12, dtype=torch.float64)


def safe_pow(base, exponent):
    """
    a debugging function used to check torch.pow for invalid arguments
    """
    # Check inputs for NaN values
    if torch.any(torch.isnan(base)) or torch.any(torch.isnan(exponent)):
        log.error("NaN values found in inputs of pow operation")
        raise ValueError

    if torch.isclose(base, zero, threshold * 0.1):
        log.debug("base is too small")
        # raise ValueError

    if base < 0:
        log.error("taking a negative base")
        raise ValueError

    # Perform the pow operation
    result = torch.pow(base, exponent)

    return result


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
    alpha_pow = safe_pow(alpha * h, n)
    outer_alpha_pow = (safe_pow(1.0 + alpha_pow, m))
    result = (
        1.0 / outer_alpha_pow * (theta_e - theta_r)
    ) + theta_r
    return error_check(result)


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
    return error_check(h_min_cm)


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
    return error_check(bc_psib)


def calc_se_from_theta(theta: Tensor, theta_e: Tensor, theta_r: Tensor) -> Tensor:
    """
    function to calculate Se from theta
    ((theta-r)/(e-r));
    :param theta_init: the calculated inital theta
    :param theta_e: theta_e
    :param theta_r: theta_r
    :return: Se this is the relative (scaled 0-1) water content, like Theta
    """
    result = (theta - theta_r)/(theta_e - theta_r)
    return error_check(result)


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
    internal_state = safe_pow(alpha * h, n)
    result = 1.0 / (safe_pow(1.0 + internal_state, m))
    return error_check(result)


def calc_k_from_se(se: Tensor, ksat: Tensor, m: Tensor) -> Tensor:
    """
    function to calculate K from Se
    (Ksat * sqrt(Se) * pow(1.0 - pow(1.0 - pow(Se,1.0/m), m), 2.0));
    :param se: this is the relative (scaled 0-1) water content, like Theta
    :param ksat: saturated hydraulic conductivity
    :param m: Van Genuchten
    :return: hydraulic conductivity (K)
    """
    se_pow = safe_pow(se, 1.0 / m)
    # If SE = 1, our gradient chain breaks since we'll be doing a .pow() of 0
    # Taking the derivative of that won't work
    base = 1.0 - se_pow
    if torch.isclose(base, zero, threshold):
        base = base + threshold
    outside_se_pow = safe_pow(base, m)
    exponent = torch.tensor(2.0)
    result = (
        ksat
        * torch.sqrt(se)
        * safe_pow(1.0 - outside_se_pow, exponent)
    )
    return error_check(result)


def calc_h_from_se(
    se: torch.Tensor, alpha: torch.Tensor, m: torch.Tensor, n: torch.Tensor
):
    """
    function to calculate h from Se using:
    1.0/alpha*pow(pow(Se,-1.0/m)-1.0,1.0/n))
    """
    se_pow = safe_pow(se, (-1.0 / m))
    base = se_pow - 1.0
    # If SE = 1, our gradient chain breaks since we'll be doing a .pow() of 0
    # Taking the derivative of that won't work
    if torch.isclose(base, zero, threshold):
        base = base + threshold
    outside_se_pow = safe_pow(base, (1.0 / n))
    result = 1.0 / alpha * outside_se_pow
    return error_check(result)


def error_check(result):
    """
    Checks to make sure there are no NaN values
    """
    if torch.any(torch.isnan(result)):
        log.error("NaN values found in result")
        raise ValueError
    else:
        return result