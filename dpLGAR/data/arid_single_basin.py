"""A file to store the function where we read the input data"""
from datetime import datetime, timedelta
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

from dpLGAR.data import BasinData
from dpLGAR.data.scaler import BasinNormScaler, MinMax
from dpLGAR.data.utils import read_df

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Basin_06332515(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # self.scaler = BasinNormScaler(cfg=self.cfg.data)
        self.scaler = MinMax(cfg=self.cfg.data)
        self.data = BasinData()
        self._read_attributes()
        self._read_forcings()
        self._read_observations()
        # stat_dict_path = Path(self.cfg.data.scaler.stat_dict)
        # if not stat_dict_path.is_file():
        #     self.scaler.create_stat_dict(self.data)
        #     self.scaler.write()
        # else:
        #     self.scaler.initialize(self.data)
        self.normalized_data = BasinData()
        self._set_normalized_data()

    def _read_attributes(self):
        df = read_df(self.cfg.data.attributes_files.polaris)
        attr_df = df[self.cfg.data.varC]
        camels_df = read_df(self.cfg.data.attributes_files.camels)
        self.data.basin_attributes = attr_df
        self.data.camels_attributes = camels_df

    def _read_forcings(self):
        precip_df = read_df(self.cfg.data.forcing_files.precip)
        filtered_precip_df = self._filter_dates(precip_df, "date")
        precip = filtered_precip_df["total_precipitation"] * self.cfg.conversions.mm_to_cm
        pet_df = read_df(self.cfg.data.forcing_files.pet)
        filtered_pet_df = self._filter_dates(pet_df, "Date")
        pet = filtered_pet_df["PET(mm/h)"] * self.cfg.conversions.mm_to_cm
        self.data.pet = pet
        self.data.precip = precip

    def _read_observations(self):
        streamflow_df = read_df(self.cfg.data.streamflow_observations)
        soil_moisture_df = read_df(self.cfg.data.soil_moisture_observations)
        filtered_streamflow_df = self._filter_dates(streamflow_df, "date")
        filtered_soil_moisture_df = self._filter_dates(soil_moisture_df, "Date")
        self.data.streamflow = filtered_streamflow_df.values
        self.data.soil_moisture = filtered_soil_moisture_df.values

    @staticmethod
    def _log_normal_transform(value):
        return torch.log10(torch.sqrt(value) + 0.1)

    def _set_normalized_data(self):
        # TODO make sure the normalization works
        # - plotting make sure there is variance
        # - after set up normalization and LSTM
        np_attr = torch.tensor(self.data.basin_attributes.values)
        self.normalized_data.basin_attributes = self.scaler(np_attr)
        torch_precip = torch.tensor(self.data.precip.values.reshape((1, 3001)))
        torch_pet = torch.tensor(self.data.pet.values.reshape((1, 3001)))
        torch_precip[torch.isnan(torch_precip)] = 0.0
        torch_pet[torch.isnan(torch_pet)] = 0.0
        self.normalized_data.precip = self._log_normal_transform(torch_precip)
        self.normalized_data.pet = self._log_normal_transform(torch_pet)
        self.normalized_data.basin_attributes[torch.isnan(self.normalized_data.basin_attributes)] = 0.0

    def _filter_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Filters the dataframe based on date range specified in the configuration.

        Args:
        - df: The Pandas DataFrame to filter.
        - date_column: The name of the column in df that contains the date information.

        Returns:
        - A filtered DataFrame.
        """
        df[date_column] = pd.to_datetime(df[date_column])
        start_datetime = datetime.strptime(self.cfg.data.start_time, '%Y-%m-%d %H:%M:%S')
        end_datetime = start_datetime + timedelta(hours=int(self.cfg.models.endtime))
        mask = (df[date_column] >= start_datetime) & (df[date_column] <= end_datetime)
        return df.loc[mask]

    def __getitem__(self, index) -> T_co:
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed data for a particular index
        """
        return (
            self.data.pet[index],
            self.data.precip[index],
            self.normalized_data.precip[index],
            self.normalized_data.pet[index],
            self.normalized_data.basin_attributes,
            self.data.streamflow[index],
            self.data.soil_moisture[index]
        )

    def __len__(self):
        """
        Method from the torch.Dataset parent class
        """
        return self.data.pet.shape[0]

    def read_oberservations(self, cfg: DictConfig):
        """
        reading observations from NLDAS forcings
        :param cfg: the DictConfig obj
        """
        obs = read_df(cfg.data.observations)
        precip = obs["total_precipitation"]
        return precip