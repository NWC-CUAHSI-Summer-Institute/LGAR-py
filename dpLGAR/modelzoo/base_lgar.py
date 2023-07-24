import logging

from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from dpLGAR.datautils import read_test_params
from dpLGAR.modelzoo.basemodel import BaseModel
log = logging.getLogger(__name__)


class BaseLGAR(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        :param soil_information: soil attributes
        """
        super(BaseLGAR, self).__init__(cfg=cfg)

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
        num_layers = self.cfg.datazoo.num_soil_layers

        self.ponded_depth_max = nn.Parameter(torch.tensor(self.cfg.data.ponded_depth_max))
        self.alpha = nn.ParameterList([])
        self.n = nn.ParameterList([])
        self.ksat = nn.ParameterList([])
        for i in range(num_layers):
            self.alpha.append(nn.Parameter(_alpha_layer[i]))
            self.n.append(nn.Parameter(_n_layer[i]))
            # Addressing Frozen Factor
            self.ksat.append(nn.Parameter(_ksat_layer[i] * self.cfg.datautils.constants.frozen_factor))



