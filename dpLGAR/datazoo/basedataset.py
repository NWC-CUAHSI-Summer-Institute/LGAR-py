"""A file to store the function where we read the input flat_files"""
import logging
from pathlib import Path
from typing import TypeVar


import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


log = logging.getLogger("datazoo.basedataset")
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class BaseDataset(Dataset):
    def __init__(self,
                 cfg: DictConfig,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 ) -> None:
        super(BaseDataset).__init__()
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")
        else:
            self.period = period

        # TODO Support reading the list of basins from a text file
        self.basins = [basin]

        self._x = {}
        self.attributes = {}
        self._y = {}

        self._load_data()

    def _load_attributes(self):
        """This function has to return the attributes in a Tensor."""
        raise NotImplementedError

    def _load_basin_data(self):
        """
        Read and filter a CSV file for specified columns and date range.
        :param cfg: the dictionary that we're reading vars from
        :return: DataFrame filtered for specified columns and date range.
        """
        raise NotImplementedError

    def _load_data(self):
        self._load_attributes()

        self._load_basin_data()

        self._load_observations()


    def _load_observations(self):
        raise NotImplementedError
        # start_date = self.cfg.data.time_interval.warmup
        # end_date = self.cfg.data.time_interval.end
        # cols = ["date", "QObs(mm/h)"]
        # data = pd.read_csv(self.cfg.data.synthetic_file, usecols=cols, parse_dates=["date"])
        # filtered_data = data.query("@start_date <= date <= @end_date")
        # precip = filtered_data["QObs(mm/h)"]
        # precip_tensor = torch.tensor(precip.to_numpy(), device=self.cfg.device)
        # nan_mask = torch.isnan(precip_tensor)
        # precip_tensor[nan_mask] = 0.0
        # self._y = precip_tensor

    def _setup_normalization(self):
        # initialize scaler dict with default center and scale values (mean and std)
        self.scaler = {
            "center": self.soil_attributes.mean(dim=[0, 1], keepdim=True),
            "scale": self.soil_attributes.std(dim=[0, 1], keepdim=True),
        }

        # check for feature-wise custom normalization
        for feature_idx, feature_specs in self.custom_normalization.items():
            for key, val in feature_specs.items():
                # check for custom treatment of the center
                if key == "centering":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["center"][..., feature_idx] = 0.0
                    elif val.lower() == "median":
                        self.scaler["center"][..., feature_idx] = torch.median(
                            self.soil_attributes[..., feature_idx]
                        )
                    elif val.lower() == "min":
                        self.scaler["center"][..., feature_idx] = torch.min(
                            self.soil_attributes[..., feature_idx]
                        )
                    elif val.lower() == "mean":
                        # do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown centering method {val}")

                # check for custom treatment of the scale
                elif key == "scaling":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["scale"][..., feature_idx] = 1.0
                    elif val == "minmax":
                        self.scaler["scale"][..., feature_idx] = torch.max(
                            self.soil_attributes[..., feature_idx]
                        ) - torch.min(self.soil_attributes[..., feature_idx])
                    elif val == "std":
                        # do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown scaling method {val}")
                else:
                    # raise ValueError to point to the correct argument names
                    raise ValueError(
                        "Unknown dict key. Use 'centering' and/or 'scaling' for each feature."
                    )

    def __getitem__(self, index) -> T_co:
        """
        Method from the torch.Dataset parent class
        :param index: the date you're iterating on
        :return: the forcing and observed flat_files for a particular index
        """
        return self._x[index], self._y[index]

    def __len__(self):
        """
        Method from the torch.Dataset parent class
        """
        return self._x.shape[0]

    def _plot_normalization(self):
        """
        A plotting function to check the attributes normalization scheme
        """
        # Plotting each attribute
        attributes = ["clay", "sand", "silt", "ph", "om"]
        for i, attribute in enumerate(attributes):
            plt.figure()
            attribute_values = self._attributes[..., i].flatten()
            plt.scatter(range(len(attribute_values)), attribute_values)
            plt.title(f"Normalized {attribute}")
            plt.show()
