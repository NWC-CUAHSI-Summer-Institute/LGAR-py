"""A file to hold all soil functions"""
import logging
import pandas as pd
import torch
from torch import Tensor

log = logging.getLogger("physics.utils")


def calc_theta_from_h(
    h: Tensor, alpha: Tensor, n: Tensor, m: Tensor, theta_e: Tensor, theta_r: Tensor
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

def calc_h_min_cm(df: pd.DataFrame, device) -> pd.DataFrame:
    """
    # /* this is the effective capillary drive after */
    # /* Morel-Seytoux et al. (1996) eqn. 13 or 15 */
    # /* psi should not be less than this value.  */
    :param df:
    :return:
    """
    bc_psib_cm = torch.tensor(df["bc_psib_cm"].to_numpy(), device=device)
    assert torch.any(0.0 < bc_psib_cm)  # checking parameter constraints
    lambda_ = torch.tensor(df["bc_lambda"].to_numpy(), device=device)
    df["h_min_cm"] = bc_psib_cm * (2.0 + 3.0 / lambda_) / (1.0 + 3.0 / lambda_)
    return df


def calc_bc_lambda_psib_cm(alpha, n) -> (torch.Tensor, torch.Tensor):
    """
    Given van Genuchten parameters calculate estimates of
    Brooks & Corey bc_lambda and bc_psib
    :param alpha: Van Genuchten parameter
    :param n: Van Genuchten parameter
    :return:
    """
    assert not torch.any(n < 1)  # van Genuchten parameter n must be greater than 1
    m_ = 1.0 - (1.0 / n)
    p_ = 1.0 + (2.0 / m_)
    bc_lambda = 2.0 / (p_ - 3.0)
    bc_psib_cm = (
        (p_ + 3.0)
        * (147.8 + 8.1 * p_ + 0.092 * p_ * p_)
        / (2.0 * alpha * p_ * (p_ - 1.0) * (55.6 + 7.4 * p_ + p_ * p_))
    )
    return bc_lambda, bc_psib_cm
