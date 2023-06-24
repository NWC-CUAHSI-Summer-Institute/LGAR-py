"""A file to store the function where we read the input data"""
import logging

from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import (
    TypeVar,
)

from lgartorch.models.physics.utils import (
    calc_theta_from_h,
    calc_bc_lambda,
    calc_bc_psib,
    calc_h_min_cm,
    calc_m,
)

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        alpha: torch.nn.ParameterList,
        n: torch.nn.ParameterList,
        ksat: torch.nn.ParameterList,
    ) -> None:
        super().__init__()

        df = self.read_df(cfg.data.forcing_file)
        self.forcing_df = df.iloc[:cfg.models.nsteps]  # cutting off at the end of nsteps

        # Convert pandas dataframe to PyTorch tensors
        precip = torch.tensor(self.forcing_df["P(mm/h)"].values, device=cfg.device)
        pet = torch.tensor(self.forcing_df["PET(mm/h)"].values, device=cfg.device)
        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        self.x = x_.transpose(0, 1)
        # Creating a time interval
        time_values = self.forcing_df["Time"].values
        self.timestep_map = {time: idx for idx, time in enumerate(time_values)}

        # Convert soils dataframe to PyTorch tensors
        self.soils_df = self.read_df(cfg.data.soil_params_file)
        texture_values = self.soils_df["Texture"].values
        self.texture_map = {texture: idx for idx, texture in enumerate(texture_values)}
        self.c = self.generate_soil_metrics(cfg, alpha, n, ksat)
        self.column_map = {
            "theta_e": 0,
            "theta_r": 1,
            "theta_wp": 2,
            "m": 3,
            "bc_lambda": 4,
            "bc_psib_cm": 5,
            "h_min_cm": 6,
        }

        # TODO FIND OBSERVATION DATA TO TRAIN AGAINST
        self.y = torch.zeros([self.x.shape[0]], device=cfg.device).unsqueeze(1)

    def __getitem__(self, index) -> T_co:
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed data for a particular index
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """
        Method from the torch.Dataset parent class
        """
        return self.x.shape[0]

    def read_df(self, file: str) -> pd.DataFrame:
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
        self,
        cfg: DictConfig,
        alpha: torch.nn.ParameterList,
        n: torch.nn.ParameterList,
        ksat_cm_per_h: torch.nn.ParameterList,
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
        theta_e = torch.tensor(self.soils_df["theta_e"], device=cfg.device)
        theta_r = torch.tensor(self.soils_df["theta_r"], device=cfg.device)
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
            m[i] = calc_m(single_n)  # Commenting out temporarily so that our test cases match
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
