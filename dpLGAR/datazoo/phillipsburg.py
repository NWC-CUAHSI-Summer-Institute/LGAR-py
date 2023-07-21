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


class Phillipsburg(BaseDataset):
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
        raise NotImplementedError

    def _load_basin_data(self) -> Tensor:
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        raise NotImplementedError