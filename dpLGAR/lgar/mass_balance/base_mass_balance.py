import logging

import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor

log = logging.getLogger(__name__)


class BaseMassBalance:
    def __init__(self):
        super().__init__()
        self.starting_volume = torch.tensor(0.0)
        self.ending_volume = torch.tensor(0.0)
        self.precip = torch.tensor(0.0)
        self.PET = torch.tensor(0.0)
        self.AET = torch.tensor(0.0)
        self.ponded_water = torch.tensor(0.0)
        self.previous_precip = torch.tensor(0.0)
        self.infiltration = torch.tensor(0.0)
        self.runoff = torch.tensor(0.0)
        self.giuh_runoff = torch.tensor(0.0)
        self.giuh_runoff_queue = torch.tensor(0.0)
        self.discharge = torch.tensor(0.0)
        self.groundwater_discharge = torch.tensor(0.0)
        self.percolation = torch.tensor(0.0)

    def _calc_mb(self):
        return (
                self.starting_volume
                + self.precip
                + self.ponded_water
                - self.runoff
                - self.AET
                - self.ponded_water
                - self.percolation
                - self.ending_volume
        )

    def add_mass(
        self,
        precip,
        infiltration,
        AET,
        percolation,
        runoff,
        giuh_runoff,
        discharge,
        PET,
        ponded_water,
        groundwater_discharge,
    ):
        self.precip = self.precip + precip
        self.infiltration = self.infiltration + infiltration
        self.AET = self.AET + AET
        self.percolation = self.percolation + percolation
        self.runoff = self.runoff + runoff
        self.giuh_runoff = self.giuh_runoff + giuh_runoff
        self.discharge = self.discharge + discharge
        self.PET = self.PET + PET
        self.ponded_water = ponded_water
        self.groundwater_discharge = (
            self.groundwater_discharge + groundwater_discharge
        )

    def reset_internal_states(self, ending_volume):
        self.precip = torch.tensor(0.0)
        self.infiltration = torch.tensor(0.0)
        self.starting_volume = ending_volume.clone()
        self.ending_volume = torch.tensor(0.0)
        self.AET = torch.tensor(0.0)
        self.percolation = torch.tensor(0.0)
        self.runoff = torch.tensor(0.0)
        self.giuh_runoff = torch.tensor(0.0)
        self.discharge = torch.tensor(0.0)
        self.PET = torch.tensor(0.0)
        self.ponded_depth = torch.tensor(0.0)
        self.ponded_water = torch.tensor(0.0)
        self.groundwater_discharge = torch.tensor(0.0)
