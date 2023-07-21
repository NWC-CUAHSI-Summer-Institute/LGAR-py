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
    def __init__(self,
                 cfg: DictConfig,
                 is_train: bool,
                 period: str,
                 basin: str = None,) -> None:
        super(BaseDataset, self).__init__(cfg=cfg,
                                         is_train=is_train,
                                         period=period,
                                         basin=basin,)

    def _load_attributes(self) -> pd.DataFrame:
        """This function has to return the attributes in a Tensor."""
        self._attributes = read_df(self.cfg.data.soil_params_file)

    def _load_basin_data(self) -> Tensor:
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        df = read_df(self.cfg.data.forcing_file)
        self.forcing_df = df.iloc[
            : self.cfg.models.nsteps
        ]  # cutting off at the end of nsteps

        # Convert pandas dataframe to PyTorch tensors
        precip = torch.tensor(self.forcing_df["P(mm/h)"].values, device=cfg.device)
        pet = torch.tensor(self.forcing_df["PET(mm/h)"].values, device=cfg.device)
        x_ = torch.stack([precip, pet])  # Index 0: Precip, index 1: PET
        x_tr = x_.transpose(0, 1)
        # Convert from mm/hr to cm/hr
        self._x = x_tr * self.cfg.conversions.mm_to_cm