from omegaconf import DictConfig
import logging
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.data.utils import generate_soil_metrics, read_df, read_test_params
from lgartorch.models.physics.GlobalParams import GlobalParams
from lgartorch.models.physics.layers.Layer import Layer

log = logging.getLogger("models.dpLGAR")


class dpLGAR(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(dpLGAR, self).__init__()

        self.cfg = cfg

        # Setting parameters
        alpha_, n_, ksat_ = read_test_params(cfg)
        self.alpha = nn.ParameterList([])
        self.n = nn.ParameterList([])
        self.ksat = nn.ParameterList([])
        for i in range(alpha_.shape[0]):
            self.alpha.append(nn.Parameter(alpha_[i]))
            self.n.append(nn.Parameter(n_[i]))
            self.ksat.append(nn.Parameter(ksat_[i]))

        # Creating tensors from config variables
        global_params = GlobalParams(cfg)

        # Creating static soil params
        self.soils_df = read_df(cfg.data.soil_params_file)
        texture_values = self.soils_df["Texture"].values
        self.texture_map = {texture: idx for idx, texture in enumerate(texture_values)}
        self.c = generate_soil_metrics(self.cfg, self.soils_df, self.alpha, self.n)
        self.cfg.data.soil_property_indexes = {
            "theta_e": 0,
            "theta_r": 1,
            "theta_wp": 2,
            "m": 3,
            "bc_lambda": 4,
            "bc_psib_cm": 5,
            "h_min_cm": 6,
        }

        # Creating initial soil layer stack
        # We're only saving a reference to the top layer as all precip, PET, and runoff deal with it
        top_layer = Layer(cfg, global_params, self.c, self.alpha, self.n, self.ksat)

    def forward(self, x) -> Tensor:
        """
        The forward function to model Precip/PET through LGAR functions
        :param x: Precip and PET forcings
        :param c: Soil Attributes
        :return:
        """
        pass
