"""A file to store the function where we read the input data"""
import logging

from omegaconf import DictConfig
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

from src.physics.LGAR.utils import (
    calc_theta_from_h,
    calc_bc_lambda_psib_cm,
    calc_h_min_cm,
)

log = logging.getLogger("data.Data")
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        device = cfg.device

        self.forcing_df = self.read_df(cfg.data.forcing_file)

        # Convert pandas dataframe to PyTorch tensors
        # TODO ADD TIMEINTERVAL SELECTION
        # Creating an index for the array (ASSUMES A TIME COLUMN)
        time_values = self.forcing_df['Time'].values
        self.time_dict = {time: idx for idx, time in enumerate(time_values)}
        self.forcing_df['Timestep'] = self.forcing_df['Time'].map(self.time_dict)
        precip = torch.tensor(self.forcing_df["P(mm/h)"].values, device=device)
        pet = torch.tensor(self.forcing_df["PET(mm/h)"].values, device=device)
        self.x = torch.zeros([2, precip.shape[0]], device=device)  # Index 0: Precip, index 1: PET
        self.x[0] = precip
        self.x[1] = pet


        # Convert soils dataframe to PyTorch tensors
        self.soils_df = self.read_df(cfg.data.soil_params_file)
        texture_values = self.soils_df["Texture"].values
        texture_mapping = {texture: idx for idx, texture in enumerate(texture_values)}
        soils_data = self.generate_soil_metrics()  # Getting all columns but the Texture column
        self.c = torch.tensor(soils_data, device=device)

    def __getitem__(self, index) -> T_co:
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed data for a particular index
        """
        return self.x[index, :], self.y[index, :]

    def __len__(self):
        """
        Method from the torch.Dataset parent class
        """
        return len(self.x)

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

    def generate_soil_metrics(self) -> np.ndarray:
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
        device = self.cfg.device
        h = torch.tensor(self.cfg.data.wilting_point_psi, device=device)  # Wilting point in cm
        alpha = torch.tensor(self.soils_df["alpha(cm^-1)"], device=device)
        n = torch.tensor(self.soils_df["n"], device=device)
        m = torch.tensor(self.soils_df["m"], device=device)
        theta_e = torch.tensor(self.soils_df["theta_e"], device=device)
        theta_r = torch.tensor(self.soils_df["theta_r"], device=device)
        self.soils_df["theta_wp"] = calc_theta_from_h(h, alpha, n, m, theta_e, theta_r)
        self.soils_df["bc_lambda"], self.soils_df["bc_psib_cm"] = calc_bc_lambda_psib_cm(alpha, n)
        bc_psib_cm = torch.tensor(self.soils_df["bc_psib_cm"].to_numpy(), device=device)
        lambda_ = torch.tensor(self.soils_df["bc_lambda"].to_numpy(), device=device)
        self.soils_df["h_min_cm"] = calc_h_min_cm(df, device)

        soils_data = self.soils_df.to_numpy()[:, 1:].astype(np.float64)  # Getting all columns but the Texture column

        return soils_data
