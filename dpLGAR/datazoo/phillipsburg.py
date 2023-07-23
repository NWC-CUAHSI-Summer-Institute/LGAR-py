import logging

from omegaconf import DictConfig
import pandas as pd
import torch
from torch import Tensor
from typing import (
    TypeVar,
)

from dpLGAR.datautils.utils import read_df
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

    def load_observations(self):
        # There are no observations from LGAR-C
        return None

    def load_phillipsburg_data(self):
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
        return read_df(self.cfg.datazoo.soil_params_file)

    def _load_attributes(self):
        """This function has to return the attributes in a Tensor."""
        self._attributes = self.load_soils_df()

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
