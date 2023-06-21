"""A file to store the function where we read the input data"""
import logging

import pandas
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from src.physics.soil_functions import calc_theta_from_h
from src.tests.sanity_checks import DataError

log = logging.getLogger("data.read_forcing")
torch.set_default_dtype(torch.float64)


def read_forcing_data(cfg: DictConfig) -> (np.ndarray, torch.Tensor, torch.Tensor):
    """
    a function to read the forcing input dataset
    :param file_path: the file we want to read
    :return:
    - time
    - precipitation
    - PET
    """
    forcing_file_path = Path(cfg.data.forcing_file)
    device = cfg.device

    # Check if forcing file exists
    if not forcing_file_path.is_file():
        log.error(f"File {forcing_file_path} doesn't exist")
        raise DataError
    df = pd.read_csv(forcing_file_path)

    # Convert pandas dataframe to PyTorch tensors
    time = df["Time"].values
    precip = torch.tensor(df["P(mm/h)"].values, dtype=torch.float64, device=device)
    pet = torch.tensor(df["PET(mm/h)"].values, dtype=torch.float64, device=device)

    return time, precip, pet


def read_soils_file(
    cfg: DictConfig, wilting_point_psi_cm: torch.Tensor
) -> pd.DataFrame:
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
    device = cfg.device
    soils_file_path = Path(cfg.data.soil_params_file)

    # Check if forcing file exists
    if not soils_file_path.is_file():
        log.error(f"File {soils_file_path} doesn't exist")
        raise DataError

    # Checking the file extension so we correctly read the file
    if soils_file_path.suffix == ".csv":
        df = pd.read_csv(soils_file_path)
    elif soils_file_path.suffix == ".dat":
        df = pd.read_csv(soils_file_path, delimiter=r"\s+", engine="python")
    else:
        log.error(f"File {soils_file_path} has an invalid type")
        raise DataError

    alpha = torch.tensor(df["alpha(cm^-1)"].to_numpy(), device=device)
    m = torch.tensor(df["m"].to_numpy(), device=device)
    n = torch.tensor(df["n"].to_numpy(), device=device)
    theta_e = torch.tensor(df["theta_e"].to_numpy(), device=device)
    theta_r = torch.tensor(df["theta_r"].to_numpy(), device=device)
    df["theta_wp"] = calc_theta_from_h(wilting_point_psi_cm, df, device)

    # Given van Genuchten parameters calculate estimates of Brooks & Corey bc_lambda and bc_psib
    assert not torch.any(n < 1) # van Genuchten parameter n must be greater than 1
    m_ = 1.0 - (1.0 / n)
    p_ = 1.0 + (2.0 / m_)
    df["bc_lambda"] = 2.0 / (p_ - 3.0)
    df["bc_psib_cm"] = (
        (p_ + 3.0)
        * (147.8 + 8.1 * p_ + 0.092 * p_ * p_)
        / (2.0 * alpha * p_ * (p_ - 1.0) * (55.6 + 7.4 * p_ + p_ * p_))
    )
    bc_psib_cm = torch.tensor(df["bc_psib_cm"].to_numpy(), device=device)

    assert torch.any(0.0 < bc_psib_cm)

    # /* this is the effective capillary drive after */
    # /* Morel-Seytoux et al. (1996) eqn. 13 or 15 */
    # /* psi should not be less than this value.  */
    lambda_ = torch.tensor(df["bc_lambda"].to_numpy(), device=device)
    df["h_min_cm"] = bc_psib_cm * (2.0 + 3.0 / lambda_) / (1.0 + 3.0 / lambda_)
    return df
