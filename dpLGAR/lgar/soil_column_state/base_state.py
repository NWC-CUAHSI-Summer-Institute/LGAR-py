from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


class BaseState:
    def __init__(self, cfg: DictConfig) -> None:
        super(BaseState).__init__()

        self.device = cfg.device

        self.layer_thickness_cm = None
        self.cum_layer_thickness = None
        self.num_layers = None
        self.soil_depth_cm = None
        self.initial_psi = None
        self.ponded_depth_max = None
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

        self.soil_index = None

        self._initialize_config_parameters(cfg)
        self._initialize_giuh_params(cfg)

        # Variables for specific functions:
        self.relative_moisture_at_which_PET_equals_AET = torch.tensor(0.75, device=self.device)
        self.nint = torch.tensor(cfg.datautils.constants.nint, device=self.device)

    def _initialize_config_parameters(self, cfg: DictConfig) -> None:
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
        self.layer_thickness_cm = torch.tensor(
            cfg.data.layer_thickness, device=cfg.device
        )
        self.cum_layer_thickness = torch.zeros(
            [len(cfg.data.layer_thickness)], device=cfg.device
        )
        self.layer_thickness_cm = torch.tensor(cfg.data.layer_thickness, device=cfg.device)
        self.cum_layer_thickness = self.layer_thickness_cm.cumsum(dim=0)
        self.num_layers = len(cfg.data.layer_thickness)
        self.soil_depth_cm = self.cum_layer_thickness[-1]

        self.initial_psi = torch.tensor(cfg.data.initial_psi, device=cfg.device)

        self.use_closed_form_G = cfg.data.use_closed_form_G

        self.num_soil_types = torch.tensor(cfg.data.max_soil_types, device=cfg.device)

        self.wilting_point_psi_cm = torch.tensor(
            cfg.data.wilting_point_psi_cm, device=cfg.device
        )
        self.frozen_factor = torch.tensor(cfg.constants.frozen_factor, device=cfg.device)
        self.soil_index = cfg.data.soil_index

    def _initialize_giuh_params(self, cfg: DictConfig):
        """
        Initalizing all giuh params
        :param cfg:
        :return:
        """
        self.giuh_ordinates = torch.tensor(cfg.data.giuh_ordinates, device=cfg.device)
        self.giuh_runoff = torch.zeros([len(self.giuh_ordinates)], device=cfg.device)
        self.num_giuh_ordinates = len(self.giuh_ordinates)
