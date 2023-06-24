from omegaconf import DictConfig
import logging
import pandas as pd
from pathlib import Path
import torch
from torch import Tensor

from lgartorch.models.physics.utils import (
    calc_theta_from_h,
    calc_bc_lambda,
    calc_bc_psib,
    calc_h_min_cm,
    calc_m,
)

log = logging.getLogger("data.utils")


def read_df(file: str) -> pd.DataFrame:
    """
    a function to read a input dataset
    :param file: the file we want to read
    :return: a pandas df
    """
    file_path = Path(file)

    # Checking the file extension so we correctly read the file
    # Csv files are usually from forcings
    # .dat are from soil
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".dat":
        df = pd.read_csv(file_path, delimiter=r"\s+", engine="python")
    else:
        log.error(f"File {file_path} has an invalid type")
        raise ValueError
    return df


def generate_soil_metrics(
    cfg: DictConfig,
    soils_df: pd.DataFrame,
    alpha: torch.nn.ParameterList,
    n: torch.nn.ParameterList,
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
        cfg.data.wilting_point_psi, device=cfg.device
    )  # Wilting point in cm
    # alpha = torch.tensor(self.soils_df["alpha(cm^-1)"], device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    # n = torch.tensor(self.soils_df["n"], device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    # m = torch.tensor(self.soils_df["m"], device=cfg.device)  # TODO We're calculating this through n
    theta_e = torch.tensor(soils_df["theta_e"], device=cfg.device)
    theta_r = torch.tensor(soils_df["theta_r"], device=cfg.device)
    # k_sat = torch.tensor(self.soils_df["Ks(cm/h)"] device=cfg.device) this is a nn.param. Commenting out to not pull from the .dat file
    # ksat_cm_per_h = k_sat * cfg.constants.frozen_factor
    m = torch.zeros(len(alpha), device=cfg.device)
    theta_wp = torch.zeros(len(alpha), device=cfg.device)
    bc_lambda = torch.zeros(len(alpha), device=cfg.device)
    bc_psib_cm = torch.zeros(len(alpha), device=cfg.device)
    h_min_cm = torch.zeros(len(alpha), device=cfg.device)
    for i in range(len(alpha)):
        single_alpha = alpha[i]
        single_n = n[i]
        m[i] = calc_m(
            single_n
        )  # Commenting out temporarily so that our test cases match
        theta_wp[i] = calc_theta_from_h(
            h, single_alpha, single_n, m[i], theta_e[i], theta_r[i]
        )
        bc_lambda[i] = calc_bc_lambda(m[i])
        bc_psib_cm[i] = calc_bc_psib(single_alpha, m[i])
        h_min_cm[i] = calc_h_min_cm(bc_lambda[i], bc_psib_cm[i])

    soils_data = torch.stack(
        [
            theta_e,
            theta_r,
            theta_wp,
            m,
            bc_lambda,
            bc_psib_cm,
            h_min_cm,
        ]
    )  # Putting all numeric columns in a tensor other than the Texture column
    return soils_data


def read_test_params(cfg: DictConfig) -> (Tensor, Tensor, Tensor):
    alpha_test_params = torch.tensor(
        [
            0.01,
            0.02,
            0.01,
            0.03,
            0.04,
            0.03,
            0.02,
            0.03,
            0.01,
            0.02,
            0.01,
            0.01,
            0.0031297,
            0.0083272,
            0.0037454,
            0.009567,
            0.005288,
            0.004467,
        ],
        device=cfg.device,
    )
    n_test_params = torch.tensor(
        [
            1.25,
            1.42,
            1.47,
            1.75,
            3.18,
            1.21,
            1.33,
            1.45,
            1.68,
            1.32,
            1.52,
            1.66,
            1.6858,
            1.299,
            1.6151,
            1.3579,
            1.5276,
            1.4585,
        ],
        device=cfg.device,
    )

    k_sat_test_params = torch.tensor(
        [
            0.612,
            0.3348,
            0.504,
            4.32,
            26.64,
            0.468,
            0.54,
            1.584,
            1.836,
            0.432,
            0.468,
            0.756,
            0.45,
            0.07,
            0.45,
            0.07,
            0.02,
            0.2,
        ],
        device=cfg.device,
    )

    return alpha_test_params, n_test_params, k_sat_test_params