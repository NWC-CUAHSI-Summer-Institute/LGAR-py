from omegaconf import DictConfig
import logging
import pandas as pd
from pathlib import Path
import torch
from torch import Tensor

from dpLGAR.modelzoo.physics.utils import (
    calc_theta_from_h,
    calc_bc_lambda,
    calc_bc_psib,
    calc_h_min_cm,
    calc_m,
)

log = logging.getLogger("flat_files.utils")


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


def read_test_params() -> (Tensor, Tensor, Tensor):
    """
    Sets the values of alpha, n, and ksat to the values defined nby the LGAR-C soils profiles
    """
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
            1.6599999999,
            1.6858,
            1.299,
            1.6151,
            1.3579,
            1.5276,
            1.4585,
        ],
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
    )

    return alpha_test_params, n_test_params, k_sat_test_params
