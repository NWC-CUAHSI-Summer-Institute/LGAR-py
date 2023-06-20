#####################################################################################
# Python Author: Tadd Bindas
# C authors : Ahmad Jan, Fred Ogden, and Peter La Follette
# year    : 2022
# email   : tkb5476@psu.edu
# - The file contains lgar subroutines
#####################################################################################

"""
LL          GGG
LL        GGG GGG       AAA  A    RR   RRRR
LL       GG    GGG    AAA AAAA     RR RR  RR
LL       GG     GG   AA     AA     RRR
LL      GGG    GGG  AAA     AA     RR
LL       GG  GG GG   AA     AAA    RR
LL        GGGG  GG    AAA  AA A    RR
LL              GG      AAAA   AA  RR
LL              GG
LLLLLLLL  GG   GG
             GGGG

The lgar specific functions based on soil physics

SKETCH SHOWING 3 SOIL LAYERS AND 4 WETTING FRONTS

theta_r                                  theta1             theta_e
  --------------------------------------------------------------   --depth_cm = 0    -------> theta
  r                                        f                   e
  r                                        f                   e
  r                                --------1                   e   --depth_cm(f1)
  r                               f         \                  e
  r     1st soil layer            f          \                 e
  r                               f wetting front number       e
  r                               f /                          e
  r                               f/                           e
  --------------------------------2-----------------------------  -- depth_cm(f2)
     r                                      f              e
     r     2nd soil layer                   f              e
     r                                      f              e
|      ---------------------------------------3---------------      -- depth_cm(f3)
|        r                                          f     e
|        r                                          f     e
|        r       3rd  soil layer                    f     e
|        r                                          f     e
V        -------------------------------------------4------         -- depth_cm(f4)
depth

############################################################################################
@param soil_moisture_wetting_fronts  : 1D array of thetas (soil moisture content) per wetting front;
                                       output to other models (e.g. soil freeze-thaw)
@param soil_depth_wetting_fronts : 1D array of absolute depths of the wetting fronts [meters];
				   output to other models (e.g. soil freeze-thaw)
############################################################################################
"""
from collections import deque
import logging
import numpy as np
from omegaconf import DictConfig
import torch

from src.data.read import read_soils_file
from src.physics.soil_functions import (
    calc_theta_from_h,
    calc_se_from_theta,
    calc_k_from_se,
)
from src.physics.WettingFront import WettingFront

log = logging.getLogger("physics.Lgar")


def seconds_to_hours(x):
    return torch.div(x, 3600.0)


def minutes_to_hours(x):
    return torch.div(x, 60.0)


def hours_to_hours(x):
    return torch.div(x, 1.0)


def seconds_to_seconds(x):
    return torch.mul(x, 1.0)


def minutes_to_seconds(x):
    return torch.mul(x, 60.0)


def hours_to_seconds(x):
    return torch.mul(x, 3600.0)


def days_to_seconds(x):
    return torch.mul(x, 86400.0)


division_switcher = {
    "s": seconds_to_hours,
    "sec": seconds_to_hours,
    "min": minutes_to_hours,
    "minute": minutes_to_hours,
    "h": hours_to_hours,
    "hr": hours_to_hours,
}

multiplication_switcher = {
    "s": seconds_to_seconds,
    "sec": seconds_to_seconds,
    "min": minutes_to_seconds,
    "minute": minutes_to_seconds,
    "h": hours_to_seconds,
    "hr": hours_to_seconds,
    "d": days_to_seconds,
    "day": days_to_seconds,
}


class LGAR:
    def __init__(self, cfg: DictConfig) -> None:
        log.debug(
            "------------- Initialization from config file ----------------------"
        )
        self.device = cfg.device

        # Setting these options to false (default)
        self.sft_coupled = False
        self.use_closed_form_G = False

        # # Setting the parameter check variables to False
        # is_layer_thickness_set = False
        # is_initial_psi_set = False
        # is_timestep_set = False
        # is_endtime_set = False
        # is_forcing_resolution_set = False
        # is_layer_soil_type_set = False
        # is_wilting_point_psi_cm_set = False
        # is_soil_params_file_set = False
        # is_max_soil_types_set = False
        # is_giuh_ordinates_set = False
        # is_soil_z_set = False
        # is_ponded_depth_max_cm_set = False

        # Setting soil parameters based on the cfg file
        self.layer_thickness_cm = torch.tensor(
            cfg.data.layer_thickness, dtype=torch.float64, device=self.device
        )
        self.cum_layer_thickness = torch.zeros(
            [len(cfg.data.layer_thickness)], dtype=torch.float64, device=self.device
        )
        self.cum_layer_thickness[0] = self.layer_thickness_cm[0]
        for i in range(1, self.cum_layer_thickness.shape[0]):
            self.cum_layer_thickness[i] = (
                self.cum_layer_thickness[i - 1].clone() + self.layer_thickness_cm[i]
            )
        self.num_layers = len(cfg.data.layer_thickness)
        self.soil_depth_cm = self.cum_layer_thickness[-1]
        # is_layer_thickness_set = True

        log.debug(f"Number of Layers: {self.num_layers}")
        [
            log.debug(f"Thickness, cum. depth {x.item()}")
            for x in self.cum_layer_thickness
        ]

        self.initial_psi = torch.tensor(
            cfg.data.initial_psi, dtype=torch.float64, device=self.device
        )
        # is_initial_psi_set = True

        timestep_unit = cfg.data.units.timestep_h[0]
        _func_ = division_switcher.get(timestep_unit, seconds_to_hours)
        self.timestep_h = _func_(
            torch.tensor(cfg.data.timestep, dtype=torch.float64, device=self.device)
        )
        assert self.timestep_h > 0
        # is_timestep_set = True

        endtime_unit = cfg.data.units.endtime_s[0]
        _func_ = multiplication_switcher.get(endtime_unit, seconds_to_seconds)
        self.endtime_s = _func_(
            torch.tensor(cfg.data.endtime, dtype=torch.float64, device=self.device)
        )
        assert self.endtime_s > 0
        # is_endtime_set = True

        forcing_unit = cfg.data.units.forcing_resolution_h[0]
        _func_ = division_switcher.get(forcing_unit, seconds_to_hours)
        self.forcing_resolution_h = _func_(
            torch.tensor(
                cfg.data.forcing_resolution, dtype=torch.float64, device=self.device
            )
        )
        assert self.forcing_resolution_h > 0
        # is_forcing_resolution_set = True

        self.ponded_depth_max_cm = torch.max(
            torch.tensor(
                cfg.data.ponded_depth_max, dtype=torch.float64, device=self.device
            )
        )
        # is_ponded_depth_max_cm_set = True

        self.use_closed_form_G = cfg.data.use_closed_form_G

        # HARDCODING A 1 SINCE PYTHON is 0-BASED FOR LISTS AND C IS 1-BASED
        self.layer_soil_type = np.array(cfg.data.layer_soil_type) - 1
        # is_layer_soil_type_set = True

        self.num_soil_types = torch.tensor(
            cfg.data.max_soil_types, dtype=torch.float64, device=self.device
        )
        # is_max_soil_types_set = True

        self.wilting_point_psi_cm = torch.tensor(
            cfg.data.wilting_point_psi, dtype=torch.float64, device=self.device
        )
        # is_wilting_point_psi_cm_set = True

        self.giuh_ordinates = torch.tensor(
            cfg.data.giuh_ordinates, dtype=torch.float64, device=self.device
        )
        self.num_giuh_ordinates = torch.tensor(
            len(cfg.data.giuh_ordinates), dtype=torch.float64, device=self.device
        )

        # Read in soils information
        self.soils_df = read_soils_file(cfg)

        self.soil_temperature = torch.tensor(0, dtype=torch.float64, device=self.device)
        self.soil_temperature_z = torch.tensor(
            0, dtype=torch.float64, device=self.device
        )
        self.num_cells_z = torch.tensor(1, dtype=torch.float64, device=self.device)

        # add 1.0e-08 to prevent truncation error
        # TODO ADD MORE TESTING TO THIS INTERVAL
        self.forcing_interval = torch.div(
            self.forcing_resolution_h, (self.timestep_h + 1.0e-08)
        )

        self.frozen_factor = torch.ones(
            int(self.num_layers), dtype=torch.float64, device=self.device
        )

        num_wetting_vars = 4
        # Creating a torch matrix that will hold wetting fronts.
        # Each wetting front is a row. Columns are depth, theta, layer, dzdt_cm_per_h
        self.wetting_fronts = deque()
        self.initialize_wetting_front()

        # Running a mass balance check
        self.calc_mass_balance()

        # initially we start with a dry surface (no surface ponding)
        self.ponded_depth_cm = torch.tensor(0, dtype=torch.float64, device=self.device)
        # No. of spatial intervals used in trapezoidal integration to compute G
        self.nint = 120  # hacked, not needed to be an input option
        self.num_wetting_fronts = self.num_layers
        self.time_s = 0.0
        self.timesteps = 0.0

        # Finish initializing variables
        self.shape = [self.num_layers, self.num_wetting_fronts]
        self.num_wetting_fronts = self.num_layers  #TODO Ask about this line? It seems stupid
        self.soil_depth_wetting_fronts = torch.zeros([self.num_wetting_fronts])
        self.soil_moisture_wetting_fronts = torch.zeros([self.num_wetting_fronts])
        for i in range(self.soil_moisture_wetting_fronts.shape[0]):
            self.soil_moisture_wetting_fronts[i] = self.wetting_fronts[i].theta
            self.soil_depth_wetting_fronts[i] = (self.wetting_fronts[i].depth_cm * cfg.units.cm_to_m)  # CONVERTING FROM CM TO M

        # Initializing the rest of the input vars
        self.precipitation_mm_per_h = torch.tensor(-1, dtype=torch.float64, device=self.device)
        self.PET_mm_per_h = torch.tensor(-1, dtype=torch.float64, device=self.device)

        # Initializing the rest of the mass balance variables to zero
        self.volprecip_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volin_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volend_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volAET_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrech_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrunoff_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrunoff_giuh_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volQ_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volPET_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volon_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volprecip_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficail wetting front
        self.volon_timestep_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.precip_previous_timestep_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        # setting flux from groundwater_reservoir_to_stream to zero, will be non-zero when groundwater reservoir is added/simulated
        self.volQ_gw_timestep_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)









    def initialize_wetting_front(self) -> None:
        """
        calculates initial theta (soil moisture content) and hydraulic conductivity
        from the prescribed psi value for each of the soil layers

        THE PARAMETERS BELOW ARE FROM THE C IMPLEMENTATION. DOCUMENTING BELOW FOR CONSISTENCY
        :parameter num_layers (int): the number of soil layers
        :parameter initial_psi_cm: the initial psi of the soil layer
        :parameter layer_soil_type (array): the type of soil per layer
        :parameter cum_layer_thickness (array): the cumulative thickness of the soil
        :parameter frozen factor (array): ???
        :parameter soil_properties (df): a dataframe of all of the soils and their properties
        :return: The wetting front per soil layer
        """

        for i in range(self.num_layers):
            soil_type = self.layer_soil_type[i]
            soil_properties = self.soils_df.iloc[soil_type]
            theta_init = calc_theta_from_h(
                self.initial_psi, soil_properties, self.device
            )
            log.debug(
                f"Layer: {i}\n"
                f"Texture: {soil_properties['Texture']}\n"
                f"Theta_init: {theta_init}\n"
                f"theta_r: {soil_properties['theta_r']}\n"
                f"theta_e: {soil_properties['theta_e']}\n"
                f"alpha(cm^-1) : {soil_properties['alpha(cm^-1)']}\n"
                f"n: {soil_properties['n']}\n"
                f"m: {soil_properties['m']}\n"
                f"Ks(cm/h): {soil_properties['Ks(cm/h)']}\n"
            )
            bottom_flag = True
            wetting_front = WettingFront(
                depth=self.cum_layer_thickness[i],
                theta=theta_init,
                layer=i,
                bottom_flag=bottom_flag,
            )
            wetting_front.psi_cm = self.initial_psi
            se = calc_se_from_theta(
                theta=wetting_front.theta,
                e=soil_properties["theta_e"],
                r=soil_properties["theta_r"],
            )
            ksat_cm_per_h = self.frozen_factor[i] * soil_properties["Ks(cm/h)"]
            wetting_front.k_cm_per_h = calc_k_from_se(
                se, ksat_cm_per_h, soil_properties["m"]
            )
            self.wetting_fronts.append(wetting_front)

    def calc_mass_balance(self):
        """
        Calculates a mass balance from your variables (Known as lgar_calc_mass_bal() in the C code)
        :return:
        """
        sum = torch.tensor(0, dtype=torch.float64)

        if len(self.wetting_fronts) == 0:
            log.info("No Wetting Fronts")
            return sum

        for i in range(len(self.wetting_fronts)):
            current = self.wetting_fronts[i]
            base_depth = (
                self.cum_layer_thickness[current.layer - 1]
                if i > 0
                else torch.tensor(0.0, dtype=torch.float64, device=self.device)
            )
            # This is not the last entry in the list
            if i < len(self.wetting_fronts) - 1:
                next = self.wetting_fronts[i + 1]
                if next.layer == current.layer:
                    sum = sum + (current.depth_cm - base_depth) * (
                        current.theta - next.theta
                    )
                else:
                    sum = sum + (current.depth_cm - base_depth) * current.theta
            else:  # This is the last entry in the list. This must be the deepest front in the final layer
                sum += current.theta * (current.depth_cm - base_depth)

        return sum

    def frozen_factor_hydraulic_conductivity(self) -> None:
        """
        calculates frozen factor based on L. Wang et al. (www.hydrol-earth-syst-sci.net/14/557/2010/)
        uses layered-average soil temperatures and an exponential function to compute frozen fraction
        for each layer
        :return: None
        """
        raise NotImplementedError