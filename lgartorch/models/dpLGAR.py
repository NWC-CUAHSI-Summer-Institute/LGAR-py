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
import time
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.data.utils import generate_soil_metrics, read_df, read_test_params
from lgartorch.models.physics.GlobalParams import GlobalParams
from lgartorch.models.physics.layers.Layer import Layer
from lgartorch.models.physics.lgar.frozen_factor import (
    frozen_factor_hydraulic_conductivity,
)

log = logging.getLogger("models.dpLGAR")


class dpLGAR(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """

        :param cfg:
        """
        super(dpLGAR, self).__init__()

        self.cfg = cfg

        # Setting NN parameters
        alpha_, n_, ksat_ = read_test_params(cfg)
        self.alpha = nn.ParameterList([])
        self.n = nn.ParameterList([])
        self.ksat = nn.ParameterList([])
        for i in range(alpha_.shape[0]):
            self.alpha.append(nn.Parameter(alpha_[i]))
            self.n.append(nn.Parameter(n_[i]))
            # Addressing Frozen Factor
            self.ksat.append(nn.Parameter(ksat_[i] * cfg.constants.frozen_factor))

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

        # Creating tensors from config variables
        self.global_params = GlobalParams(cfg)

        # Creating initial soil layer stack
        # We're only saving a reference to the top layer as all precip, PET, and runoff deal with it
        layer_index = 0  # This is the top layer
        self.top_layer = Layer(
            self.global_params,
            layer_index,
            self.c,
            self.alpha,
            self.n,
            self.ksat,
            self.texture_map,
        )

        # Running the initial mass balance check
        self.volstart_cm = self.calc_mass_balance()

        # Setting output tracking params
        self.precip_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.PET_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.AET_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.volend_timestep_cm = self.volstart_cm.clone()
        self.volin_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficail wetting front
        self.volon_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.precip_previous_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        # self.volrunoff_timestep_cm = torch.tensor(0.0, device=self.device)
        # self.volrech_timestep_cm = torch.tensor(0.0, device=self.device)
        self.surface_runoff_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.volrunoff_giuh_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.volQ_timestep_cm = torch.tensor(0.0, device=self.cfg.device)
        self.volQ_gw_timestep_cm = torch.tensor(0.0, device=self.cfg.device)

        # Variables we want to save at every timestep
        self.volrunoff_timestep_cm = torch.zeros(
            [self.cfg.models.nsteps], device=self.cfg.device
        )
        self.volrech_timestep_cm = torch.zeros(
            [self.cfg.models.nsteps], device=self.cfg.device
        )

    def forward(self, x) -> Tensor:
        """
        The forward function to model Precip/PET through LGAR functions
        /* Note unit conversion:
        Pr and PET are rates (fluxes) in mm/h
        Pr [mm/h] * 1h/3600sec = Pr [mm/3600sec]
        Model timestep (dt) = 300 sec (5 minutes for example)
        convert rate to amount
        Pr [mm/3600sec] * dt [300 sec] = Pr[mm] * 300/3600.
        in the code below, subtimestep_h is this 300/3600 factor (see initialize from config in lgar.cxx)
        :param x: Precip and PET forcings
        :return: runoff to be used for validation
        """
        # TODO implement the LGAR functions for if there is precip or PET
        precip = x[0][0]
        pet = x[0][1]
        previous_precip = torch.tensor(0.0, device=self.cfg.device)
        if self.global_params.sft_coupled:
            # TODO work in frozen soil components
            frozen_factor_hydraulic_conductivity()
        for i in tqdm(range(self.cfg.models.nsteps), desc="Running dpLGAR"):
            for j in range(self.cfg.models.num_subcycles):
                precip_subtimestep = precip * self.cfg.subcycle_length_h
                ponded_depth_subtimestep = precip_subtimestep + self.volon_timestep_cm
                pet_subtimestep = pet * self.cfg.subcycle_length_h
                # Determining wetting cases
                create_surficial_front = self.create_surficial_front(
                    previous_precip, precip_subtimestep
                )
                is_top_layer_saturated = self.top_layer.is_saturated()
                if pet_subtimestep > 0.0:
                    raise NotImplementedError
                    # self.top_layer.calc_aet(pet_subtimestep)
                else:
                    AET_subtimestep_cm = torch.tensor(0.0, device=self.device)
                if create_surficial_front:
                    if is_top_layer_saturated:
                        # It's raining
                        raise NotImplementedError
                        # self.top_layer.input_precip(precip_subtimestep)
                else:
                    if ponded_depth_subtimestep > 0:
                        #  infiltrate water based on the infiltration capacity given no new wetting front
                        #  is created and that there is water on the surface (or raining).
                        raise NotImplementedError
                    else:
                        if ponded_depth_subtimestep < self.global_params.ponded_depth_max_cm:
                            raise NotImplementedError
                        else:
                            raise NotImplementedError
                    # move wetting fronts if no new wetting front is created. Otherwise, movement
                    # of wetting fronts has already happened at the time of creating surficial front,
                    # so no need to move them here. */
                    raise NotImplementedError
                if pet_subtimestep > 0:
                    self.top_layer.calc_aet(pet_subtimestep)
                previous_precip = precip_subtimestep

            time.sleep(0.001)

        return self.volrunoff_timestep_cm

    def calc_mass_balance(self) -> Tensor:
        """
        Calculates a mass balance from your variables (Known as lgar_calc_mass_bal() in the C code)
        This is a recursive stack function. It calls the top of the stack,
        and will go until the bottom is reach
        :return: Sum
        """
        return self.top_layer.mass_balance()

    def create_surficial_front(self, previous_precip, precip_subtimestep):
        """
        Checks the volume of water on the surface, and if it's raining or has recently rained
        to determine if any water has infiltrated the surface
        :param previous_precip:
        :param precip_subtimestep:
        :param volon_timestep_cm:
        :return:
        """
        # This enusures we don't add extra mass from a previous storm event
        has_previous_precip = (previous_precip == 0.0).item()
        is_it_raining = (precip_subtimestep > 0.0).item()
        is_there_ponded_water = self.volon_timestep_cm == 0
        return has_previous_precip and is_it_raining and is_there_ponded_water
