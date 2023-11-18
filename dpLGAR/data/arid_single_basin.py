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

from dpLGAR.data.scaler import BasinNormScaler
from dpLGAR.data.utils import read_df

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Basin_06332515(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scaler = BasinNormScaler(cfg=self.cfg.data)
        self.c = None
        self.x = None
        self.y = None
        self._read_attributes()
        self._read_forcings()
        self._read_observations()
        stat_dict_path = Path(self.cfg.data.scaler.stat_dict)
        if not stat_dict_path.is_file():
            self.scaler.create_stat_dict((self.c, self.x), self.y)
            self.scaler.write()
        else:
            self.scaler.initialize(data)

    def _read_attributes(self):
        df = read_df(self.cfg.data.attributes_file)
        attr_df = df[self.cfg.data.attributes]
        self.c = attr_df.values

    def _read_forcings(self):
        df = read_df(self.cfg.data.forcing_file)
        filtered_df = self._filter_dates(df, "date")
        filtered_df["potential_evaporation"] = filtered_df["potential_evaporation"] * self.cfg.conversions.mm_to_cm
        filtered_df["total_precipitation"] = filtered_df["total_precipitation"] * self.cfg.conversions.mm_to_cm
        self.x = filtered_df[self.cfg.data.forcings].values
        # precip = torch.tensor(forcing_df["P(mm/h)"].values)
        # pet = torch.tensor(forcing_df["PET(mm/h)"].values)
        # x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        # x_tr = x_.transpose(0, 1)
        # self.x = x_tr * self.cfg.conversions.mm_to_cm # Convert from mm/hr to cm/hr

    def _read_observations(self):
        streamflow_df = read_df(self.cfg.data.streamflow_observations)
        soil_moisture_df = read_df(self.cfg.data.soil_moisture_observations)
        filtered_streamflow_df = self._filter_dates(streamflow_df, "date")
        filtered_soil_moisture_df = self._filter_dates(soil_moisture_df, "Date")

        self.y = (filtered_streamflow_df.values, filtered_soil_moisture_df.values)

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

    def _calculate_PET(self):
        # Get forcing
        self.nldas_forcing = nldas_forcing
        self.input_forcing = self.prepare_input(nldas_forcing)

        # Get CAMELS basin attributes
        basin_attrs = pd.read_csv(self.cfg.camels_attr_file)
        basin_attrs['gauge_id'] = basin_attrs['gauge_id'].astype(str).str.zfill(8)
        basin_idx = basin_attrs['gauge_id'] == basin_id
        self.lon = basin_attrs['gauge_lon'][basin_idx].values[0]
        self.lat = basin_attrs['gauge_lat'][basin_idx].values[0]
        self.elevation = basin_attrs['elev_mean'][basin_idx].values[0]


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

    def read_oberservations(self, cfg: DictConfig):
        """
        reading observations from NLDAS forcings
        :param cfg: the DictConfig obj
        """
        obs = read_df(cfg.data.observations)
        precip = obs["total_precipitation"]
        return precip