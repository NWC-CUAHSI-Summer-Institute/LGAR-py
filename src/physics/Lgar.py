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
    calc_h_from_se,
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

        # Defining all of the variables that are created in the `initialize()` functions
        self.layer_thickness_cm = None
        self.cum_layer_thickness = None
        self.num_layers = None
        self.soil_depth_cm = None
        self.initial_psi = None
        self.ponded_depth_max_cm = None
        self.layer_soil_type = None
        self.num_soil_types = None
        self.wilting_point_psi_cm = None
        self.giuh_ordinates = None
        self.num_giuh_ordinates = None

        self.timestep_h = None
        self.endtime_s = None
        self.forcing_resolution_h = None

        self.soils_df = None
        self.soil_temperature = None
        self.soil_temperature_z = None
        self.num_cells_z = None
        self.forcing_interval = None
        self.frozen_factor = None
        self.wetting_fronts = None

        self.ponded_depth_cm = None
        self.nint = None
        self.num_wetting_fronts = None
        self.time_s = None
        self.timesteps = None
        self.shape = None
        self.volprecip_cm = None
        self.volin_cm = None
        self.volend_cm = None
        self.volAET_cm = None
        self.volrech_cm = None
        self.volrunoff_cm = None
        self.volrunoff_giuh_cm = None
        self.volQ_cm = None
        self.volon_cm = None
        self.volprecip_cm = None
        self.volon_timestep_cm = None
        self.precip_previous_timestep_cm = None
        self.volQ_gw_timestep_cm = None

        self.soil_depth_wetting_fronts = None
        self.soil_moisture_wetting_fronts = None
        self.precipitation_mm_per_h = None
        self.PET_mm_per_h = None

        # Setting these options to false (default)
        self.sft_coupled = False
        self.use_closed_form_G = False

        # Setting variables
        self.initialize_config_parameters(cfg)
        self.initialize_time_parameters(cfg)
        self.initialize_wetting_front(cfg)

        # Running a mass balance check
        self.calc_mass_balance()

        self.initialize_starting_parameters(cfg)

        # Creating a pointer to the correct wetting front index
        self.current = 0

    def initialize_config_parameters(self, cfg: DictConfig) -> None:
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

        log.debug(f"Number of Layers: {self.num_layers}")
        [
            log.debug(f"Thickness, cum. depth {x.item()}")
            for x in self.cum_layer_thickness
        ]

        self.initial_psi = torch.tensor(
            cfg.data.initial_psi, dtype=torch.float64, device=self.device
        )
        self.ponded_depth_max_cm = torch.max(
            torch.tensor(
                cfg.data.ponded_depth_max, dtype=torch.float64, device=self.device
            )
        )

        self.use_closed_form_G = cfg.data.use_closed_form_G

        # HARDCODING A 1 SINCE PYTHON is 0-BASED FOR LISTS AND C IS 1-BASED
        self.layer_soil_type = np.array(cfg.data.layer_soil_type) - 1

        self.num_soil_types = torch.tensor(
            cfg.data.max_soil_types, dtype=torch.float64, device=self.device
        )

        self.wilting_point_psi_cm = torch.tensor(
            cfg.data.wilting_point_psi, dtype=torch.float64, device=self.device
        )

        self.giuh_ordinates = torch.tensor(
            cfg.data.giuh_ordinates, dtype=torch.float64, device=self.device
        )
        self.num_giuh_ordinates = torch.tensor(
            len(cfg.data.giuh_ordinates), dtype=torch.float64, device=self.device
        )

    def initialize_time_parameters(self, cfg: DictConfig) -> None:
        timestep_unit = cfg.data.units.timestep_h[0]
        _func_ = division_switcher.get(timestep_unit, seconds_to_hours)
        self.timestep_h = _func_(
            torch.tensor(cfg.data.timestep, dtype=torch.float64, device=self.device)
        )
        assert self.timestep_h > 0

        endtime_unit = cfg.data.units.endtime_s[0]
        _func_ = multiplication_switcher.get(endtime_unit, seconds_to_seconds)
        self.endtime_s = _func_(
            torch.tensor(cfg.data.endtime, dtype=torch.float64, device=self.device)
        )
        assert self.endtime_s > 0

        forcing_unit = cfg.data.units.forcing_resolution_h[0]
        _func_ = division_switcher.get(forcing_unit, seconds_to_hours)
        self.forcing_resolution_h = _func_(
            torch.tensor(
                cfg.data.forcing_resolution, dtype=torch.float64, device=self.device
            )
        )
        assert self.forcing_resolution_h > 0

    def initialize_wetting_front(self, cfg: DictConfig) -> None:
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
        # Read in soils information
        self.soils_df = read_soils_file(cfg)

        self.soil_temperature = torch.tensor(0, dtype=torch.float64, device=self.device)
        self.soil_temperature_z = torch.tensor(
            0, dtype=torch.float64, device=self.device
        )
        self.num_cells_z = torch.tensor(1, dtype=torch.float64, device=self.device)

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
                layer_num=i,
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
                self.cum_layer_thickness[current.layer_num - 1]
                if i > 0
                else torch.tensor(0.0, dtype=torch.float64, device=self.device)
            )
            # This is not the last entry in the list
            if i < len(self.wetting_fronts) - 1:
                next = self.wetting_fronts[i + 1]
                if next.layer_num == current.layer_num:
                    sum = sum + (current.depth_cm - base_depth) * (
                        current.theta - next.theta
                    )
                else:
                    sum = sum + (current.depth_cm - base_depth) * current.theta
            else:  # This is the last entry in the list. This must be the deepest front in the final layer
                sum += current.theta * (current.depth_cm - base_depth)

        return sum

    def initialize_starting_parameters(self, cfg: DictConfig) -> None:
        # initially we start with a dry surface (no surface ponding)
        self.ponded_depth_cm = torch.tensor(0, dtype=torch.float64, device=self.device)
        # No. of spatial intervals used in trapezoidal integration to compute G
        self.nint = 120  # hacked, not needed to be an input option
        self.num_wetting_fronts = self.num_layers
        self.time_s = 0.0
        self.timesteps = 0.0

        # Finish initializing variables
        self.shape = [self.num_layers, self.num_wetting_fronts]
        self.num_wetting_fronts = (
            self.num_layers
        )  # TODO Ask about this line? It seems stupid
        self.soil_depth_wetting_fronts = torch.zeros([self.num_wetting_fronts])
        self.soil_moisture_wetting_fronts = torch.zeros([self.num_wetting_fronts])
        for i in range(self.soil_moisture_wetting_fronts.shape[0]):
            self.soil_moisture_wetting_fronts[i] = self.wetting_fronts[i].theta
            self.soil_depth_wetting_fronts[i] = (
                self.wetting_fronts[i].depth_cm * cfg.units.cm_to_m
            )  # CONVERTING FROM CM TO M

        # Initializing the rest of the input vars
        self.precipitation_mm_per_h = torch.tensor(
            -1, dtype=torch.float64, device=self.device
        )
        self.PET_mm_per_h = torch.tensor(-1, dtype=torch.float64, device=self.device)

        # Initializing the rest of the mass balance variables to zero
        self.volprecip_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volin_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volend_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volAET_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrech_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrunoff_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volrunoff_giuh_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )
        self.volQ_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volPET_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volon_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.volprecip_cm = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficail wetting front
        self.volon_timestep_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )
        self.precip_previous_timestep_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )

        # setting flux from groundwater_reservoir_to_stream to zero, will be non-zero when groundwater reservoir is added/simulated
        self.volQ_gw_timestep_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )

    def frozen_factor_hydraulic_conductivity(self) -> None:
        """
        calculates frozen factor based on L. Wang et al. (www.hydrol-earth-syst-sci.net/14/557/2010/)
        uses layered-average soil temperatures and an exponential function to compute frozen fraction
        for each layer
        :return: None
        """
        raise NotImplementedError

    def wetting_front_free_drainage(self) -> torch.Tensor:
        """
        Following the code of LGAR-C to determine the wetting fronts free of drainage
        :return:
        """
        wf_that_supplies_free_drainage_demand = 0
        max_index = len(self.wetting_fronts)
        while self.current < max_index:
            front = self.wetting_fronts[self.current]
            if (self.current + 1) < max_index:
                if front.layer_num == self.wetting_fronts[self.current + 1].layer_num:
                    break
                else:
                    wf_that_supplies_free_drainage_demand = (
                        wf_that_supplies_free_drainage_demand + 1
                    )
            self.current = self.current + 1

        if wf_that_supplies_free_drainage_demand > self.num_wetting_fronts:
            wf_that_supplies_free_drainage_demand = (
                wf_that_supplies_free_drainage_demand - 1
            )

        self.current = 0  # Reset index

        log.debug(
            f"wetting_front_free_drainage = {wf_that_supplies_free_drainage_demand}"
        )
        return wf_that_supplies_free_drainage_demand

    def theta_mass_balance(
        self,
        layer_num,
        soil_num,
        psi_cm,
        new_mass,
        prior_mass,
        delta_thetas,
        delta_thickness,
        soil_properties,
    ):
        """
        The function does mass balance for a wetting front to get an updated theta.
        The head (psi) value is iteratively altered until the error between prior mass and new mass
        is within a tolerance.
        :param layer_num:
        :param soil_num:
        :param psi_cm:
        :param new_mass:
        :param prior_mass:
        :param delta_thetas:
        :param delta_thickness:
        :param soil_properties:
        :return:
        """
        psi_cm_loc = psi_cm
        delta_mass = torch.abs(
            new_mass - prior_mass
        )  # mass difference between the new and prior
        tolerance = 1e-12

        factor = 1.0
        switched = False  # flag that determines capillary head to be incremented or decremented

        theta = torch.tensor(
            0.0, device=self.device
        )  # this will be updated and returned
        psi_cm_loc_prev = psi_cm_loc
        delta_mass_prev = delta_mass
        count_no_mass_change = torch.tensor(0.0, device=self.device)
        break_no_mass_change = torch.tensor(5.0, device=self.device)

        # check if the difference is less than the tolerance
        if delta_mass <= tolerance:
            theta = calc_theta_from_h(psi_cm_loc, soil_properties, self.device)
            return theta

        # the loop increments/decrements the capillary head until mass difference between
        # the new and prior is within the tolerance
        while delta_mass > tolerance:
            if new_mass > prior_mass:
                psi_cm_loc = psi_cm_loc + 0.1 * factor
                switched = False
            else:
                if not switched:
                    switched = True
                    factor = factor * 0.1

                psi_cm_loc_prev = psi_cm_loc
                psi_cm_loc = psi_cm_loc - 0.1 * factor

                if psi_cm_loc < 0 and psi_cm_loc_prev != 0:
                    psi_cm_loc = psi_cm_loc_prev * 0.1

            theta = calc_theta_from_h(psi_cm_loc, soil_properties, self.device)
            mass_layers = delta_thickness[layer_num] * (theta - delta_thetas[layer_num])

            for i in range(layer_num):
                soil_num_loc = self.layer_soil_type[
                    i
                ]  # _loc denotes the variable is local to the loop
                soil_properties_loc = self.soils_df.iloc[soil_num_loc]
                theta_layer = calc_theta_from_h(
                    psi_cm_loc, soil_properties_loc, self.device
                )

                mass_layers = mass_layers + delta_thickness[i] * (
                    theta_layer - delta_thetas[i]
                )

            new_mass = mass_layers
            delta_mass = torch.abs(new_mass - prior_mass)

            # stop the loop if the error between the current and previous psi is less than 10^-15
            # 1. enough accuracy, 2. the algorithm can't improve the error further,
            # 3. avoid infinite loop, 4. handles a corner case when prior mass is tiny (e.g., <1.E-5)
            # printf("A1 = %.20f, %.18f %.18f %.18f %.18f \n ",fabs(psi_cm_loc - psi_cm_loc_prev) , psi_cm_loc, psi_cm_loc_prev, factor, delta_mass);

            if torch.abs(psi_cm_loc - psi_cm_loc_prev) < 1e-15 and factor < 1e-13:
                break

            if torch.abs(delta_mass - delta_mass_prev) < 1e-15:
                count_no_mass_change += 1
            else:
                count_no_mass_change = 0

            if count_no_mass_change == break_no_mass_change:
                break

            if psi_cm_loc <= 0 and psi_cm_loc_prev < 1e-50:
                break

            delta_mass_prev = delta_mass
        return theta

    def merge_wetting_fronts(self):
        """
        the function merges wetting fronts; called from self.move_wetting_fronts()
        :return:
        """
        log.debug(f"Merging wetting fronts...")
        for i in range(1, len(self.wetting_fronts)):
            log.debug(f"Merge | ********* Wetting Front = {i} *********")
            current = i - 1
            next = i
            next_to_next = i + 1 if i + 1 < len(self.wetting_fronts) else None

            # case : wetting front passing another wetting front within a layer
            if (
                self.wetting_fronts[current].depth_cm
                > self.wetting_fronts[next].depth_cm
                and self.wetting_fronts[current].layer_num
                == self.wetting_fronts[next].layer_num
                and not self.wetting_fronts[next].to_bottom
            ):
                current_mass_this_layer = self.wetting_fronts[current].depth_cm * (
                    self.wetting_fronts[current].theta - self.wetting_fronts[next].theta
                ) + self.wetting_fronts[next].depth_cm * (
                    self.wetting_fronts[next].theta
                    - self.wetting_fronts[next_to_next].theta
                )
                self.wetting_fronts[current].depth_cm = current_mass_this_layer / (
                    self.wetting_fronts[current].theta
                    - self.wetting_fronts[next_to_next].theta
                )

                layer_num = self.wetting_fronts[current].layer_num
                soil_num = self.layer_soil_type[layer_num]
                soil_properties = self.soils_df.iloc[soil_num]
                theta_e = torch.tensor(
                    soil_properties["theta_e"], dtype=torch.float64, device=self.device
                )
                theta_r = torch.tensor(
                    soil_properties["theta_r"], dtype=torch.float64, device=self.device
                )
                alpha = torch.tensor(
                    soil_properties["alpha(cm^-1)"],
                    dtype=torch.float64,
                    device=self.device,
                )
                m = torch.tensor(
                    soil_properties["m"], dtype=torch.float64, device=self.device
                )
                n = torch.tensor(
                    soil_properties["n"], dtype=torch.float64, device=self.device
                )
                se = calc_se_from_theta(
                    self.wetting_fronts[current].theta, theta_e, theta_r
                )

                ksat_cm_per_h = (
                    soil_properties["Ksat_cm_per_h"]
                    * self.frozen_factor[self.wetting_fronts[current].layer_num]
                )

                self.wetting_fronts[current].psi_cm = calc_h_from_se(se, alpha, m, n)
                self.wetting_fronts[current].k_cm_per_h = calc_k_from_se(
                    se, ksat_cm_per_h, m
                )

                log.debug(f"Deleting wetting front (before)... {i}")

                self.wetting_fronts.pop(
                    i
                )  # equivalent to listDeleteFront(next['front_num'])

                log.debug("Deleting wetting front (after)...")

        log.debug("State after merging wetting fronts...")
        for i in range(len(self.wetting_fronts)):
            self.wetting_fronts[i].print()

    def wetting_fronts_cross_layer_boundary(self):
        raise NotImplementedError

    def move_wetting_fronts(
        self,
        timestep_h,
        volin_cm,
        wf_free_drainage_demand,
        old_mass,
        num_layers,
        AET_demand_cm,
    ):
        """
        the function moves wetting fronts, merge wetting fronts and does the mass balance correction when needed
        @param current : wetting front pointing to the current node of the current state
        @param next    : wetting front pointing to the next node of the current state
        @param previous    : wetting front pointing to the previous node of the current state
        @param current_old : wetting front pointing to the current node of the previous state
        @param next_old    : wetting front pointing to the next node of the previous state
        @param head : pointer to the first wetting front in the list of the current state

        Note: '_old' denotes the wetting_front or variables at the previous timestep (or state)
        :param subtimestep_h:
        :param volin_subtimestep_cm:
        :param wf_free_drainage_demand:
        :param volend_subtimestep_cm:
        :param num_layers:
        :param AET_subtimestep_cm:
        :return:
        """
        column_depth = self.cum_layer_thickness[-1]
        previous = self.current
        current = self.current

        precip_mass_to_add = volin_cm  # water to be added to the soil
        bottom_boundary_flux_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )  # water leaving the system through the bottom boundary
        volin_cm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )  # assuming that all the water can fit in, if not then re-assign the left over water at the end

        number_of_wetting_fronts = len(self.wetting_fronts) - 1  # Indexed at 0
        last_wetting_front_index = number_of_wetting_fronts
        for i in reversed(range(len(self.wetting_fronts))):
            """
            /* ************************************************************ */
            // main loop advancing all wetting fronts and doing the mass balance
            // loop goes over deepest to top most wetting front
            // wf denotes wetting front
            """
            if i == 0 and number_of_wetting_fronts > 0:
                current = i
                next = i + 1
                previous = None

                current_old = i
                next_old = i + 1
            elif i < number_of_wetting_fronts:
                current = i
                next = i + 1
                previous = i - 1

                current_old = i
                next_old = i + 1
            elif i == number_of_wetting_fronts:
                current = i
                next = None
                previous = i - 1

                current_old = i
                next_old = None

            layer_num = self.wetting_fronts[current].layer_num
            soil_num = self.layer_soil_type[layer_num]
            soil_properties = self.soils_df.iloc[soil_num]
            theta_e = torch.tensor(
                soil_properties["theta_e"], dtype=torch.float64, device=self.device
            )
            theta_r = torch.tensor(
                soil_properties["theta_r"], dtype=torch.float64, device=self.device
            )
            alpha = torch.tensor(
                soil_properties["alpha(cm^-1)"], dtype=torch.float64, device=self.device
            )
            m = torch.tensor(
                soil_properties["m"], dtype=torch.float64, device=self.device
            )
            n = torch.tensor(
                soil_properties["n"], dtype=torch.float64, device=self.device
            )

            layer_num_above = (
                layer_num if i == 0 else self.wetting_fronts[previous].layer_num
            )
            layer_num_below = (
                layer_num + 1
                if i == last_wetting_front_index
                else self.wetting_fronts[next].layer_num
            )

            log.debug(
                f"Layers (current, above, below) == {layer_num} {layer_num_above} {layer_num_below} \n"
            )
            free_drainage_demand = torch.tensor(
                0.0, dtype=torch.float64, device=self.device
            )
            actual_ET_demand = AET_demand_cm

            if i < last_wetting_front_index and layer_num_below != layer_num:
                """// case to check if the wetting front is at the interface, i.e. deepest wetting front within a layer
                // psi of the layer below is already known/updated, so we that psi to compute the theta of the deepest current layer
                // todo. this condition can be replace by current->to_depth = FALSE && l<last_wetting_front_index
                /*             _____________
                   layer_above             |
                                        ___|
                           |
                               ________|____    <----- wetting fronts at the interface have same psi value
                               |
                   layer current    ___|
                                   |
                               ____|________
                   layer_below     |
                              _____|________
                */
                /*************************************************************************************/
                """
                next_psi_cm = self.wetting_fronts[next].psi_cm
                log.debug(
                    f"case (deepest wetting front within layer) : layer_num {layer_num} != layer_num_below {layer_num_below}"
                )
                self.wetting_fronts[current].theta = calc_theta_from_h(
                    next_psi_cm, soil_properties, self.device
                )
                self.wetting_fronts[current].psi_cm = next_psi_cm
            if (
                i == number_of_wetting_fronts
                and layer_num_below != layer_num
                and number_of_wetting_fronts == (num_layers - 1)
            ):
                """
                // case to check if the number of wetting fronts are equal to the number of layers, i.e., one wetting front per layer
                  /*************************************************************************************/
                  /* For example, 3 layers and 3 wetting fronts in a state. psi profile is constant, and theta profile is non-uniform due
                  to different van Genuchten parameters
                    theta profile       psi profile  (constant head)
                   _____________       ______________
                             |                   |
                   __________|__       __________|___
                       |                     |
                   ________|____       __________|___
                       |                         |
                   ____|________       __________|___
                */
                """
                log.debug(
                    f"case (number_of_wetting_fronts equal to num_layers) : l {i} == num_layers (0-base) {num_layers - 1} == num_wetting_fronts{number_of_wetting_fronts}"
                )

                # this is probably not needed, as dz / dt = 0 for the deepest wetting front
                self.wetting_fronts[current].depth_cm = self.wetting_fronts[
                    current
                ].depth_cm + (self.wetting_fronts[current].dzdt_cm_per_h * timestep_h)

                delta_thetas = torch.zeros([num_layers], device=self.device)
                delta_thickness = torch.zeros([num_layers], device=self.device)
                psi_cm_old = self.wetting_fronts[current_old].psi_cm

                psi_cm = self.wetting_fronts[current].psi_cm

                # mass = delta(depth) * delta(theta)
                prior_mass = (
                    self.wetting_fronts[current_old].depth_cm
                    - self.cum_layer_thickness[layer_num - 1]
                ) * (
                    self.wetting_fronts[current_old].theta - 0.0
                )  # 0.0 = next_old->theta
                new_mass = (
                    self.wetting_fronts[current].depth_cm
                    - self.cum_layer_thickness[layer_num - 1]
                ) * (
                    self.wetting_fronts[current].theta - 0.0
                )  # 0.0 = next->theta;

                for j in range(0, number_of_wetting_fronts):
                    soil_num_ = self.layer_soil_type[j]
                    soil_properties_ = self.soils_df.iloc[soil_num_]

                    # using psi_cm_old for all layers because the psi is constant across layers in this particular case
                    theta_old = calc_theta_from_h(
                        psi_cm_old, soil_properties_, self.device
                    )
                    theta_below_old = torch.tensor(
                        0.0, dtype=torch.float64, device=self.device
                    )
                    local_delta_theta_old = theta_old - theta_below_old
                    if j == 0:
                        layer_thickness = self.cum_layer_thickness[j]
                    else:
                        layer_thickness = (
                            self.cum_layer_thickness[j]
                            - self.cum_layer_thickness[j - 1]
                        )

                    prior_mass = prior_mass + layer_thickness * local_delta_theta_old

                    theta = calc_theta_from_h(psi_cm, soil_properties_, self.device)
                    theta_below = 0.0

                    new_mass = new_mass + layer_thickness * (theta - theta_below)
                    delta_thetas[j] = theta_below
                    delta_thickness[j] = layer_thickness

                delta_thickness[layer_num] = (
                    self.wetting_fronts[current].depth_cm
                    - self.cum_layer_thickness[layer_num - 1]
                )
                free_drainage_demand = 0

                if wf_free_drainage_demand == i:
                    prior_mass = (
                        prior_mass
                        + precip_mass_to_add
                        - (free_drainage_demand + actual_ET_demand)
                    )

                # theta mass balance computes new theta that conserves the mass; new theta is assigned to the current wetting front
                theta_new = self.theta_mass_balance(
                    layer_num,
                    soil_num,
                    psi_cm,
                    new_mass,
                    prior_mass,
                    delta_thetas,
                    delta_thickness,
                    soil_properties,
                )

                self.wetting_fronts[current].theta = torch.minimum(theta_new, theta_e)
                se = calc_se_from_theta(
                    self.wetting_fronts[current].theta, theta_e, theta_r
                )
                self.wetting_fronts[current].psi_cm = calc_h_from_se(se, alpha, m, n)
            if i < last_wetting_front_index and layer_num == layer_num_below:
                raise NotImplementedError
            if i == 0:
                """
                // if f_p (predicted infiltration) causes theta > theta_e, mass correction is needed.
                // depth of the wetting front is increased to close the mass balance when theta > theta_e.
                // l == 0 is the last iteration (top most wetting front), so do a check on the mass balance)
                // this part should be moved out of here to a subroutine; add a call to that subroutine
                """
                soil_num_k1 = self.layer_soil_type[wf_free_drainage_demand]
                theta_e_k1 = self.soils_df.iloc[soil_num_k1]["theta_e"]

                wf_free_drainage = self.wetting_fronts[wf_free_drainage_demand]
                mass_timestep = (old_mass + precip_mass_to_add) - (
                    actual_ET_demand + free_drainage_demand
                )

                # Making sure that the mass is correct
                assert old_mass > 0.0

                if torch.abs(wf_free_drainage.theta - theta_e_k1) < 1e-15:
                    current_mass = self.calc_mass_balance(self.cum_layer_thickness)

                    mass_balance_error = torch.abs(
                        current_mass - mass_timestep
                    )  # mass error

                    factor = torch.tensor(1.0, device=self.device)
                    switched = False
                    tolerance = 1e-12

                    # check if the difference is less than the tolerance
                    if mass_balance_error <= tolerance:
                        pass
                        # return current_mass

                    depth_new = wf_free_drainage.depth_cm

                    # loop to adjust the depth for mass balance
                    while torch.abs(mass_balance_error - tolerance) > 1e-12:
                        if current_mass < mass_timestep:
                            depth_new = (
                                depth_new
                                + torch.tensor(0.01, device=self.device) * factor
                            )
                            switched = False
                        else:
                            if not switched:
                                switched = True
                                factor = factor * torch.tensor(
                                    0.001, device=self.device
                                )
                            depth_new = depth_new - (
                                torch.tensor(0.01, device=self.device) * factor
                            )

                        wf_free_drainage.depth_cm = depth_new

                        current_mass = self.calc_mass_balance(self.cum_layer_thickness)
                        mass_balance_error = torch.abs(current_mass - mass_timestep)

        """
          // **************************** MERGING WETTING FRONT ********************************
  
          /* In the python version of LGAR, wetting front merging, layer boundary crossing, and lower boundary crossing
             all occur in a loop that happens after wetting fronts have been moved. This prevents the model from crashing,
             as there are rare but possible cases where multiple merging / layer boundary crossing events will happen in
             the same time step. For example, if two wetting fronts cross a layer boundary in the same time step, it will
             be necessary for merging to occur before layer boundary crossing. So, LGAR-C now approaches merging in the
             same way as in LGAR-Py, where wetting fronts are moved, then a new loop does merging for all wetting fronts,
             then a new loop does layer boundary corssing for all wetting fronts, then a new loop does merging again for
             all wetting fronts, and then a new loop does lower boundary crossing for all wetting fronts. this is a thorough
             way to deal with these scenarios. */
        """
        # Merge
        self.merge_wetting_fronts()

        #Cross
        self.wetting_fronts_cross_layer_boundary()

        #Merge
        self.merge_wetting_fronts()

