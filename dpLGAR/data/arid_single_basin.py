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

from dpLGAR.data.utils import read_df

log = logging.getLogger("data.Data")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Basin_06332515(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        df = read_df(cfg.data.forcing_file)
        df['date'] = pd.to_datetime(df['date'])
        start_datetime = datetime.strptime(cfg.data.start_time, '%Y-%m-%d %H:%M:%S')
        end_datetime = start_datetime + timedelta(hours=int(cfg.models.endtime))
        mask = (df['date'] >= start_datetime) & (df['date'] <= end_datetime)
        self.forcing_df = df.loc[mask]
        precip = torch.tensor(self.forcing_df["P(mm/h)"].values, device=cfg.device)
        pet = torch.tensor(self.forcing_df["PET(mm/h)"].values, device=cfg.device)
        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        x_tr = x_.transpose(0, 1)
        # Convert from mm/hr to cm/hr
        self.x = x_tr * cfg.conversions.mm_to_cm

        obs = read_df(cfg.data.observations)

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