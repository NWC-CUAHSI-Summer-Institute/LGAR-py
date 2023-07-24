from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


class BaseState:
    def __init__(self, cfg: DictConfig, ponded_depth_max: torch.Tensor) -> None:
        super(BaseState).__init__()

        self.cfg = cfg

        self.ponded_depth_max = ponded_depth_max.clone()

        self.layer_thickness_cm = None
        self.cum_layer_thickness = None
        self.num_layers = None
        self.soil_depth_cm = None
        self.initial_psi = None
        self.num_soil_types = None
        self.wilting_point_psi_cm = None
        self.giuh_ordinates = None
        self.num_giuh_ordinates = None
        self.giuh_runoff = None

        self.timestep_h = None
        self.endtime_s = None
        self.forcing_resolution_h = None

        self.frozen_factor = None

        # Setting these options to false (default)
        self.sft_coupled = False
        self.use_closed_form_G = False

        self.soil_index = self.cfg.datazoo.soil_parameter_index

        self._initialize_config_parameters()
        self._initialize_giuh_params()

        # Variables for specific functions:
        self.relative_moisture_at_which_PET_equals_AET = torch.tensor(0.75)
        self.nint = torch.tensor(self.cfg.datautils.constants.nint)

    def _initialize_config_parameters(self) -> None:
        """
        Reading variables from the config file specific to each testcase
        :param cfg: The config file
        :return: None

        Here are a list of the variables created:
        - layer_thickness_cm: the thickness of each soil layer
        - cum_layer_thickness: A list of the cumulative depth as we traverse through the soil layers
        - num_layers: The number of soil layers
        - soil_depth_cm
        - initial_psi
        - ponded_depth_max_cm
        - layer_soil_type
        - num_soil_types
        - wilting_point_psi_cm
        - layer_thickness_cm
        - giuh_ordinates
        - num_giuh_ordinates
        """
        self.layer_thickness_cm = torch.tensor(self.cfg.datazoo.layer_thickness)
        self.cum_layer_thickness = torch.zeros([len(self.cfg.datazoo.layer_thickness)])
        self.layer_thickness_cm = torch.tensor(self.cfg.datazoo.layer_thickness)
        self.cum_layer_thickness = self.layer_thickness_cm.cumsum(dim=0)
        self.num_layers = len(self.cfg.datazoo.layer_thickness)
        self.soil_depth_cm = self.cum_layer_thickness[-1]

        self.initial_psi = torch.tensor(self.cfg.datazoo.initial_psi)

        self.use_closed_form_G = self.cfg.datazoo.use_closed_form_G

        self.num_soil_types = self.cfg.datazoo.num_soil_layers

        self.wilting_point_psi_cm = torch.tensor(self.cfg.datazoo.wilting_point_psi_cm)
        self.frozen_factor = torch.tensor(self.cfg.datautils.constants.frozen_factor)
        self.soil_index = self.cfg.datazoo.soil_parameter_index

    def _initialize_giuh_params(self):
        """
        Initalizing all giuh params
        :param cfg:
        :return:
        """
        self.giuh_ordinates = torch.tensor(self.cfg.datazoo.giuh_ordinates)
        self.giuh_runoff = torch.zeros([len(self.giuh_ordinates)])
        self.num_giuh_ordinates = len(self.giuh_ordinates)
