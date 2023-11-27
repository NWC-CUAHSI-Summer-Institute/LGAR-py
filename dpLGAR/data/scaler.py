import logging

import numpy as np
from omegaconf import DictConfig
from sklearn import preprocessing
import torch

from dpLGAR.data import BasinData, Scaler

log = logging.getLogger(__name__)


class BasinNormScaler(Scaler):
    def __init__(self, cfg: DictConfig) -> None:
        super(BasinNormScaler, self).__init__(cfg=cfg)

    def create_stat_dict(self, data: BasinData):
        self.basin_area = data.camels_attributes["area_gages2"][0]
        self.mean_prep = data.camels_attributes["p_mean"][0]
        self.create_data_scaler(data.basin_attributes.values, (data.precip.values, data.pet.values))

    def initialize(self, data):
        self.basin_area = data.camels_attributes["area_gages2"][0]
        self.mean_prep = data.camels_attributes["p_mean"][0]
        self.read()


    def create_data_scaler(
        self,
        attributes: np.ndarray,
        forcings: np.ndarray,
    ) -> None:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        for k in range(len(self.cfg.varT)):
            var = self.cfg.varT[k]
            forcings_ = forcings[k]
            if var == "total_precipitation":
                self.stat_dict[var] = self.calStatgamma(forcings_)
            else:
                self.stat_dict[var] = self.calStat(forcings_)
        for k in range(len(self.cfg.varC)):
            attributes_ = attributes[k]
            var = self.cfg.varC[k]
            self.stat_dict[var] = self.calStat(attributes_)
        log.debug("Finished scaler")

    def create_obs_scaler(
        self,
        observations: torch.Tensor,
        col: str
    ) -> None:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        tmp = self.basinNorm(observations, toNorm=True)
        self.stat_dict[col] = self.calStatgamma(tmp)


class MinMax:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.scalar = None
        self.setup_normalization()

    def __call__(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        x_trans = attributes.transpose(1, 0)
        x_tensor = torch.zeros(x_trans.shape)
        for i in range(0, x_trans.shape[0]):
            x_tensor[i, :] = torch.tensor(
                self.scalar.fit_transform(x_trans[i, :].reshape(-1, 1).cpu()).transpose(
                    1, 0
                )
            )
        return torch.transpose(x_tensor, 1, 0)

    def setup_normalization(self) -> None:
        self.scalar = preprocessing.MinMaxScaler()
