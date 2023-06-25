"""
        dd         ppp      LL          GGG
        dd      ppp  ppp    LL        GGG GGG       AAA  A    RR   RRRR
        dd     pp     ppp   LL       GG    GGG    AAA AAAA     RR RR  RR
        dd     pp     pp    LL       GG     GG   AA     AA     RRR
        dd     pp    pp     LL      GGG    GGG  AAA     AA     RR
    dddddd     pp  pp       LL       GG  GG GG   AA     AAA    RR
   dd   dd     pppp         LL        GGGG  GG    AAA  AA A    RR
  dd    dd     pp           LL              GG      AAAA   AA  RR
  dd    dd     pp           LL              GG
   ddddddd     pp           LLLLLLLL  GG   GG
                                       GGGG
"""
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
        """

        :param cfg:
        """
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
            # Addressing Frozen Factor
            self.ksat.append(nn.Parameter(ksat_[i] * cfg.constants.frozen_factor))

        # Creating tensors from config variables
        # TODO make sure to add the mapping functionality
        self.global_params = GlobalParams(cfg)

        # Creating static soil params
        self.soils_df = read_df(cfg.data.soil_params_file)
        texture_values = self.soils_df["Texture"].values
        self.texture_map = {idx: texture for idx, texture in enumerate(texture_values)}
        self.c = generate_soil_metrics(self.cfg, self.soils_df, self.alpha, self.n)
        self.cfg.data.soil_property_indexes = {
            "theta_r": 0,
            "theta_e": 1,
            "theta_wp": 2,
            "theta_init": 3,
            "m": 4,
            "bc_lambda": 5,
            "bc_psib_cm": 6,
            "h_min_cm": 7,
        }

        # Creating initial soil layer stack
        # We're only saving a reference to the top layer as all precip, PET, and runoff deal with it
        layer_index = 0  # This is the top layer
        self.top_layer = Layer(
            cfg,
            self.global_params,
            layer_index,
            self.c,
            self.alpha,
            self.n,
            self.ksat,
            self.texture_map,
        )

        # Running the initial mass balance check
        self.starting_volume = self.calc_mass_balance()

    def forward(self, x) -> Tensor:
        """
        The forward function to model Precip/PET through LGAR functions
        :param x: Precip and PET forcings
        :param c: Soil Attributes
        :return:
        """
        pass

    def calc_mass_balance(self) -> Tensor:
        """
        Calculates a mass balance from your variables (Known as lgar_calc_mass_bal() in the C code)
        This is a recursive stack function. It calls the top of the stack,
        and will go until the bottom is reach
        :return: Sum
        """
        return self.top_layer.mass_balance()

