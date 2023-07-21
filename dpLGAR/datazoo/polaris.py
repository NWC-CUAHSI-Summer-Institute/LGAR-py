"""A file to store the function where we read the input flat_files"""
import logging
from pathlib import Path

from omegaconf import DictConfig
import numpy as np
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


class Polaris(BaseDataset):
    def __init__(self,
                 cfg: DictConfig,
                 is_train: bool,
                 period: str,
                 basin: str = None,) -> None:
        super(BaseDataset, self).__init__(cfg,
                                         is_train,
                                         period,
                                         basin,)

    def _load_attributes(self):
        """This function has to return the attributes in a Tensor."""
        return self.get_polaris_atributes()

    def _load_basin_data(self) -> Tensor:
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        return self.load_forcings()

    def get_polaris_atributes(self):
        # TODO support a data dir with different basin IDs
        file_name = self.cfg.data.attributes_file
        # Load the txt flat_files into a DataFrame
        df = pd.read_csv(file_name)

        # Filter columns for soil %, Ph, and organic_matter
        clay_columns = [col for col in df.columns if col.startswith("clay")]
        sand_columns = [col for col in df.columns if col.startswith("sand")]
        silt_columns = [col for col in df.columns if col.startswith("silt")]
        ph_columns = [col for col in df.columns if col.startswith("ph")]
        organic_matter_columns = [col for col in df.columns if col.startswith("om")]

        # Create a numpy array from the columns
        clay_data = df[clay_columns].values
        sand_data = df[sand_columns].values
        silt_data = df[silt_columns].values
        ph_data = df[ph_columns].values
        organic_matter_data = df[organic_matter_columns].values

        # Shape (<num_points>, <num_layers>, <num_attributes>)
        soil_attributes = torch.stack(
            [
                torch.from_numpy(clay_data),
                torch.from_numpy(sand_data),
                torch.from_numpy(silt_data),
                torch.from_numpy(ph_data),
                torch.from_numpy(organic_matter_data),
            ],
            dim=-1,
        )

        return soil_attributes
