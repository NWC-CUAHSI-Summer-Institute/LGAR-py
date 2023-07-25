import logging

from omegaconf import DictConfig
import pandas as pd
import torch
from torch import Tensor
from typing import (
    TypeVar,
)

from dpLGAR.datazoo.basedataset import BaseDataset

log = logging.getLogger("datazoo.lgar_c")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Phillipsburg(BaseDataset):
    """
    A test case for the Phillipsburg water shed
    """

    def __init__(
        self,
        cfg: DictConfig,
        is_train: bool,
        period: str,
        basin: str = None,
    ) -> None:
        super(Phillipsburg, self).__init__(
            cfg=cfg,
            is_train=is_train,
            period=period,
            basin=basin,
        )

    def _load_attributes(self):
        """This function has to return the attributes in a Tensor."""
        self.attributes = self.load_soils_df()

    def _load_basin_data(self):
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        self._x = self.load_phillipsburg_data()

    def _load_observations(self):
        # There are no observations when comparing to an LGAR-C Case
        self._y = self.load_observations()

    def load_observations(self):
        cols = ["date", "QObs(mm/h)"]
        data = pd.read_csv(self.cfg.datazoo.observations_file, usecols=cols, parse_dates=["date"])
        nsteps = int(self.cfg.datazoo.endtime)
        filtered_data = data.iloc[:nsteps]
        q_obs = filtered_data["QObs(mm/h)"]
        q_obs_tensor = torch.tensor(q_obs.to_numpy(), device=self.cfg.device)
        nan_mask = torch.isnan(q_obs_tensor)
        q_obs_tensor[nan_mask] = 0.0
        return q_obs_tensor

    def load_phillipsburg_data(self):
        from dpLGAR.datazoo import read_df
        # Configuring timesteps
        endtime_s = self.cfg.datazoo.endtime * self.cfg.datautils.conversions.hr_to_sec
        forcing_resolution_h = (
            self.cfg.datazoo.forcing_resolution
            / self.cfg.datautils.conversions.hr_to_sec
        )
        time_per_step = forcing_resolution_h * self.cfg.datautils.conversions.hr_to_sec
        nsteps = int(endtime_s / time_per_step)
        df = read_df(self.cfg.datazoo.forcing_file)
        # cutting off at the end of nsteps
        self.forcing_df = df.iloc[:nsteps]
        # Convert pandas dataframe to PyTorch tensors
        precip = torch.tensor(self.forcing_df["P(mm/h)"].values, device=self.cfg.device)
        pet = torch.tensor(self.forcing_df["PET(mm/h)"].values, device=self.cfg.device)
        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        x_tr = x_.transpose(0, 1)
        # Convert from mm/hr to cm/hr
        return x_tr * self.cfg.datautils.conversions.mm_to_cm

    def load_soils_df(self):
        from dpLGAR.datazoo import read_df
        return read_df(self.cfg.datazoo.soil_params_file)
