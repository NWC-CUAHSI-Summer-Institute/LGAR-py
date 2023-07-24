import logging

import pandas as pd
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from dpLGAR.datautils import read_test_params
from dpLGAR.datautils.utils import generate_soil_metrics
from dpLGAR.modelzoo.basemodel import BaseModel

log = logging.getLogger(__name__)


class BaseLGAR(BaseModel):
    def __init__(self, cfg: DictConfig, c: pd.DataFrame) -> None:
        """

        :param cfg:
        :param soil_information: soil attributes
        """
        super(BaseLGAR, self).__init__(cfg=cfg, c=c)

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

    def forward(self, i, x) -> (Tensor, Tensor):
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
        raise NotImplementedError

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

