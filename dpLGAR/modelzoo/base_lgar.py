import logging

import pandas as pd
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from dpLGAR.datautils import read_test_params
from dpLGAR.datautils.utils import generate_soil_metrics
from dpLGAR.lgar.mass_balance.local_mass_balance import LocalMassBalance
from dpLGAR.lgar.lgar import run
from dpLGAR.modelzoo.basemodel import BaseModel

log = logging.getLogger(__name__)


class BaseLGAR(BaseModel):
    def __init__(self, cfg: DictConfig, c: pd.DataFrame) -> None:
        """

        :param cfg:
        :param soil_information: soil attributes
        """
        super(BaseLGAR, self).__init__(cfg=cfg, c=c)
        subtimestep_mb = None

    def _create_soil_params(self, c: pd.DataFrame):
        soil_types = self.cfg.datazoo.layer_soil_type
        self.theta_e = torch.tensor(c["theta_e"])[soil_types]
        self.theta_r = torch.tensor(c["theta_r"])[soil_types]
        self.soil_parameters = generate_soil_metrics(
            self.cfg,
            self.alpha,
            self.n,
            self.theta_e,
            self.theta_r,
        )

    def _set_parameters(self):
        self.set_parameter_lists()

    def forward(self, x) -> (Tensor, Tensor):
        """
        The forward function to model Precip/PET through LGAR functions
        /* Note unit conversion:
        Pr and PET are rates (fluxes) in mm/h
        Pr [mm/h] * 1h/3600sec = Pr [mm/3600sec]
        Model timestep (dt) = 300 sec (5 minutes for example)
        convert rate to amount
        Pr [mm/3600sec] * dt [300 sec] = Pr[mm] * 300/3600.
        in the code below, subtimestep_h is this 300/3600 factor (see initialize from config in lgar.cxx)
        :param i: The current timestep index
        :param x: Precip and PET forcings
        :return: runoff to be used for validation
        """
        # TODO FIX INDEXING
        precip = x[0][0][1]
        pet = x[0][0][1]
        self.subtimestep_mb = LocalMassBalance()
        self.subtimestep_mb.groundwater_discharge = torch.tensor(0.0)
        self.local_mb.bottom_boundary_flux = torch.tensor(0.0, device=self.cfg.device)
        self.subtimestep_mb = self.local_mb.ending_volume.clone()
        # TODO add frozen factor
        # if self.global_params.sft_coupled:
        #     frozen_factor_hydraulic_conductivity()
        subtimestep_h = self.cfg.modelzoo.subcycle_length * (1 / self.cfg.datautils.conversions.hr_to_sec)
        num_subcycles = int(
            self.cfg.modelzoo.forcing_resolution / self.cfg.modelzoo.subcycle_length
        )
        runoff = torch.zeros([int(num_subcycles)])
        for i in range(num_subcycles):
            runoff[i] = run(
                self,
                precip,
                pet,
                subtimestep_h,
            )
        return runoff

    def set_parameter_lists(self):
        """
        Setting the starting values of the differentiable LGAR model
        """
        _alpha, _n, _ksat = read_test_params()
        soil_types = self.cfg.datazoo.layer_soil_type
        _alpha_layer = _alpha[soil_types]
        _n_layer = _n[soil_types]
        _ksat_layer = _ksat[soil_types]

        self.ponded_depth_max = nn.Parameter(
            torch.tensor(self.cfg.datazoo.ponded_depth_max, dtype=torch.float64)
        )
        self.alpha = nn.ParameterList([nn.Parameter(a) for a in _alpha_layer])
        self.n = nn.ParameterList([nn.Parameter(n) for n in _n_layer])
        self.ksat = nn.ParameterList([
            nn.Parameter(k * self.cfg.datautils.constants.frozen_factor)
            for k in _ksat_layer
        ])

    def print_params(self):
        for i in range(len(self.alpha)):
            alpha = self.alpha[i]
            log.info(f"Alpha for soil {i + 1}: {alpha.detach().item():.4f}")
        for i in range(len(self.n)):
            n = self.n[i]
            log.info(f"n for soil {i + 1}: {n.detach().item():.4f}")
        for i in range(len(self.ksat)):
            ksat = self.ksat[i]
            log.info(f"Ksat for soil {i + 1}: {ksat.detach().item():.4f}")
        for i in range(len(self.theta_e)):
            theta_e = self.theta_e[i]
            log.info(f"theta_e for soil {i + 1}: {theta_e.detach().item():.4f}")
        for i in range(len(self.theta_r)):
            theta_r = self.theta_r[i]
            log.info(f"theta_r for soil {i + 1}: {theta_r.detach().item():.4f}")
        log.info(f"Max Ponded Depth: {self.ponded_depth_max.detach().item():.4f}")

