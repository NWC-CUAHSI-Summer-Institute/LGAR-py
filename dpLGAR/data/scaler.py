import logging

from omegaconf import DictConfig
import torch

from dpLGAR.data import Scaler

log = logging.getLogger(__name__)


class BasinNormScaler(Scaler):
    def __init__(self, cfg: DictConfig) -> None:
        super(BasinNormScaler, self).__init__(cfg=cfg)

    def create_stat_dict(self,  data, observations):
        self.basin_area = data.areas.reshape(-1, 1)
        self.mean_prep = data.p_mean.reshape(-1, 1)
        self.create_obs_scaler(observations.observations)
        self.create_data_scaler(data.attributes, data.forcings)

    def create_data_scaler(
        self,
        attributes: torch.Tensor,
        forcings: torch.Tensor,
    ) -> None:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        for k in range(len(self.cfg.varT)):
            var = self.cfg.varT[k]
            if var == "prcp":
                self.stat_dict[var] = self.calStatgamma(forcings[:, :, k])
            else:
                self.stat_dict[var] = self.calStat(forcings[:, :, k])
        for k in range(len(self.cfg.varC)):
            var = self.cfg.varC[k]
            self.stat_dict[var] = self.calStat(attributes[:, k])
        log.debug("Finished scaler")

    def create_obs_scaler(
        self,
        observations: torch.Tensor,
    ) -> None:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        tmp = self.basinNorm(observations, toNorm=True)
        self.stat_dict["runoff"] = self.calStatgamma(tmp)
