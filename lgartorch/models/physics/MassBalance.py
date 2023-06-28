from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger("models.physics.lgar.green_ampt")


class MassBalance:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.device = cfg.device

        self.precip = torch.tensor(0.0, device=self.device)
        self.infiltration = torch.tensor(0.0, device=self.device)
        self.starting_volume = torch.tensor(0.0, device=self.device)
        self.ending_volume = torch.tensor(0.0, device=self.device)
        self.AET = torch.tensor(0.0, device=self.device)
        self.percolation = torch.tensor(0.0, device=self.device)
        self.runoff = torch.tensor(0.0, device=self.device)
        self.giuh_runoff = torch.tensor(0.0, device=self.device)
        self.discharge = torch.tensor(0.0, device=self.device)
        self.PET = torch.tensor(0.0, device=self.device)
        self.ponded_depth = torch.tensor(0.0, device=self.device)

        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficail wetting front
        self.ponded_water = torch.tensor(0.0, device=self.device)

        # setting flux from groundwater_reservoir_to_stream to zero, will be non-zero when groundwater reservoir is added/simulated
        self.groundwater_discharge = torch.tensor(0.0, device=self.device)

    def change_mass(self, model):
        self.precip = self.precip + model.precip
        self.infiltration = self.infiltration + model.infiltration
        self.starting_volume = self.starting_volume + model.self.precip + model.precip
        self.ending_volume = self.ending_volume + model.ending_volume
        self.AET = self.AET + model.AET
        self.percolation = self.percolation + model.percolation
        self.runoff = self.runoff + model.runoff
        self.giuh_runoff = self.giuh_runoff + model.giuh_runoff
        self.discharge = self.discharge + model.discharge
        self.PET = self.PET + model.PET
        self.ponded_depth = self.ponded_depth + model.ponded_depth
        self.ponded_water = self.ponded_water + model.ponded_water
        self.groundwater_discharge = (
            self.groundwater_discharge + model.groundwater_discharge
        )

        model.precip = torch.tensor(0.0, device=self.device)
        model.PET = torch.tensor(0.0, device=self.device)
        model.AET = torch.tensor(0.0, device=self.device)
        model.ending_volume = self.starting_volume.clone()
        model.ponded_water = torch.tensor(0.0, device=self.device)
        model.precip_previous_timestep_cm = torch.tensor(0.0, device=self.device)
        model.infiltration = torch.tensor(0.0, device=self.device)
        model.surface_runoff_timestep_cm = torch.tensor(0.0, device=self.device)
        model.giuh_runoff = torch.tensor(0.0, device=self.device)
        model.discharge = torch.tensor(0.0, device=self.device)
        model.groundwater_discharge = torch.tensor(0.0, device=self.device)
        model.percolation = torch.tensor(0.0, device=self.device)
        model.runoff = torch.tensor(0.0, device=self.device)

    def report_mass(self, global_params):
        for i in range(global_params.num_giuh_ordinates):
            self.giuh_runoff = self.giuh_runoff + global_params.giuh_runoff[i]

        global_error_cm = (
            self.starting_volume
            + self.precip
            - self.runoff
            - self.AET
            - self.ponded_water
            - self.percolation
            - self.ending_volume
        )

        log.info("********************************************************* ")
        log.info("-------------------- Simulation Summary ----------------- ")
        log.info("------------------------ Mass balance ------------------- ")
        log.info(f"Initial water in soil    = {self.starting_volume}14.10f cm")
        log.info(f"Total precipitation      = {self.precip}14.10f cm")
        log.info(f"Total infiltration       = {self.infiltration}14.10f cm")
        log.info(f"Final water in soil      = {self.ending_volume}14.10f cm")
        log.info(f"Surface ponded water     = {self.ponded_water}14.10f cm")
        log.info(f"Surface runoff           = {self.runoff}14.10f cm")
        log.info(f"GIUH runoff              = {self.giuh_runoff}14.10f cm")
        log.info(f"Total percolation        = {self.percolation}14.10f cm")
        log.info(f"Total AET                = {self.AET}14.10f cm")
        log.info(f"Total PET                = {self.PET}14.10f cm")
        log.info(f"Total discharge (Q)      = {self.discharge}14.10f cm")
        log.info(f"Global balance           =   {global_error_cm}.6e cm")
