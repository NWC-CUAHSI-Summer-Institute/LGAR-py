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

from dpLGAR.data.utils import read_df

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Data(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.x = self.get_forcings(cfg)

        self.soil_information = self.get_attributes(cfg)

        self.y = self.get_observations(cfg)
        # self.y = torch.rand([self.x.shape[0]], device=cfg.device)

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

    def get_forcings(self, cfg: DictConfig):
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        file_path = cfg.data.forcing_file
        start_date = cfg.data.time_interval.warmup
        end_date = cfg.data.time_interval.end
        cols = ['date', 'potential_evaporation', 'total_precipitation']
        data = pd.read_csv(file_path, usecols=cols, parse_dates=['date'])
        filtered_data = data.query('@start_date <= date <= @end_date')
        # Unit is kg/m^2
        precip = torch.tensor(filtered_data["total_precipitation"].to_numpy(), device=cfg.device)
        # Units is kg/m^2
        PET = torch.tensor(filtered_data["potential_evaporation"].to_numpy(), device=cfg.device)
        stacked_forcings = torch.stack([precip, PET])
        # Note: kg/m^2 == mm, so we need to convert to CM
        x = stacked_forcings.transpose(0,1) * cfg.conversions.mm_to_cm
        return x

    def get_basin_area(self, cfg):
        """
        Read and filter a CSV file for a specified basin id to get the basin area.
        :param cfg: the DictConfig obj

        :return: Basin area for the specified basin id.
        """
        file_path = cfg.data.area_file
        basin_id = cfg.data.basin_id
        data = pd.read_csv(file_path)
        formatted_basin_id = f"Gage-{basin_id}"
        filtered_data = data[data['gauge_id'] == formatted_basin_id]
        return filtered_data['AREA_sqkm'].values[0] if not filtered_data.empty else None

    def get_attributes(self, cfg: DictConfig):
        """
        Reading attributes from the soil params file
        """
        file_name = cfg.data.soil_params_file
        basin_id = cfg.data.basin_id
        # Load the txt data into a DataFrame
        data = pd.read_csv(file_name, sep=';')
        data["gauge_id"] = data["gauge_id"].astype('str').str.zfill(8)
        # Filter the DataFrame for the specified basin id
        filtered_data = data[data['gauge_id'] == basin_id]
        soil_depth = filtered_data["soil_depth_statsgo"].item() * cfg.conversions.m_to_cm
        soil_texture = filtered_data["soil_texture_class"].item()
        soil_index = filtered_data["soil_index"].item()
        return [soil_depth, soil_texture, soil_index]

    def get_observations(self, cfg: DictConfig):
        """
        reading observations from NLDAS forcings
        :param cfg: the DictConfig obj
        """
        obs = read_df(cfg.data.observations_file)
        precip = obs["QObs(mm/h)"]
        precip_tensor = torch.tensor(precip.to_numpy(), device=cfg.device)
        nan_mask = torch.isnan(precip_tensor)
        # Filling NaNs with 0 as there is no streamflow
        precip_tensor[nan_mask] = 0.0
        return precip_tensor