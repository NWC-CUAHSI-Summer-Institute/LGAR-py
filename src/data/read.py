"""A file to store the function where we read the input data"""
import logging

import pandas
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from src.tests.sanity_checks import DataError

log = logging.getLogger("data.read_forcing")


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

    forcings = torch.stack([precip, pet])
    x = torch.transpose(forcings, 0, 1)

    return time, x


def read_soils_file(cfg: DictConfig) -> pd.DataFrame:
    """
    Reading the soils dataframe
    :param cfg: the config file

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
    soils_file_path = Path(cfg.data.soil_params_file)

    # Check if forcing file exists
    if not soils_file_path.is_file():
        log.error(f"File {soils_file_path} doesn't exist")
        raise DataError

    # Checking the file extension so we correctly read the file
    if soils_file_path.suffix == '.csv':
        df = pd.read_csv(soils_file_path)
    elif soils_file_path.suffix == '.dat':
        df = pd.read_csv(soils_file_path, delimiter=r'\s+', engine='python')
    else:
        log.error(f"File {soils_file_path} has an invalid type")
        df = None
    return df

