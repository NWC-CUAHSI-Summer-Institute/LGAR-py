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
from src.physics.LGAR.utils import (
    calc_theta_from_h,
    calc_se_from_theta,
    calc_h_from_se,
    calc_k_from_se,
)
from src.physics.wetting_fronts.WettingFront import WettingFront
from src.physics.LGAR.utils import read_soils
from src.physics.LGAR.geff import calc_geff

log = logging.getLogger("physics.Lgar")
torch.set_default_dtype(torch.float64)


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
        self.prev_wetting_fronts = None

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
        self.volQ_cm = None

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

        # Initializing initial states:
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
            cfg.data.layer_thickness, device=self.device
        )
        self.cum_layer_thickness = torch.zeros(
            [len(cfg.data.layer_thickness)], device=self.device
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

        self.initial_psi = torch.tensor(cfg.data.initial_psi, device=self.device)
        self.ponded_depth_max_cm = torch.max(
            torch.tensor(cfg.data.ponded_depth_max, device=self.device)
        )

        self.use_closed_form_G = cfg.data.use_closed_form_G

        # HARDCODING A 1 SINCE PYTHON is 0-BASED FOR LISTS AND C IS 1-BASED
        self.layer_soil_type = np.array(cfg.data.layer_soil_type) - 1

        self.num_soil_types = torch.tensor(cfg.data.max_soil_types, device=self.device)

        self.wilting_point_psi_cm = torch.tensor(
            cfg.data.wilting_point_psi, device=self.device
        )

        self.giuh_ordinates = torch.tensor(cfg.data.giuh_ordinates, device=self.device)
        self.num_giuh_ordinates = len(cfg.data.giuh_ordinates)

    def initialize_time_parameters(self, cfg: DictConfig) -> None:
        timestep_unit = cfg.data.units.timestep_h[0]
        _func_ = division_switcher.get(timestep_unit, seconds_to_hours)
        self.timestep_h = _func_(torch.tensor(cfg.data.timestep, device=self.device))
        assert self.timestep_h > 0

        endtime_unit = cfg.data.units.endtime_s[0]
        _func_ = multiplication_switcher.get(endtime_unit, seconds_to_seconds)
        self.endtime_s = _func_(torch.tensor(cfg.data.endtime, device=self.device))
        assert self.endtime_s > 0

        forcing_unit = cfg.data.units.forcing_resolution_h[0]
        _func_ = division_switcher.get(forcing_unit, seconds_to_hours)
        self.forcing_resolution_h = _func_(
            torch.tensor(cfg.data.forcing_resolution, device=self.device)
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
        self.soils_df = read_soils_file(cfg, self.wilting_point_psi_cm)

        self.soil_temperature = torch.tensor(0, device=self.device)
        self.soil_temperature_z = torch.tensor(0, device=self.device)
        self.num_cells_z = torch.tensor(1, device=self.device)

        # TODO ADD MORE TESTING TO THIS INTERVAL
        self.forcing_interval = torch.div(
            self.forcing_resolution_h, (self.timestep_h + 1.0e-08)
        )

        self.frozen_factor = torch.ones(int(self.num_layers), device=self.device)

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
            log.debug(f"Layer: {i}")
            log.debug(f"Texture: {soil_properties['Texture']}")
            log.debug(f"Theta_init: {theta_init.item()}")
            log.debug(f"theta_r: {soil_properties['theta_r'].item()}")
            log.debug(f"theta_e: {soil_properties['theta_e'].item()}")
            log.debug(f"alpha(cm^-1) : {soil_properties['alpha(cm^-1)'].item()}")
            log.debug(f"n: {soil_properties['n'].item()}")
            log.debug(f"m: {soil_properties['m'].item()}")
            log.debug(f"Ks(cm/h): {soil_properties['Ks(cm/h)'].item()}")
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
            if current.layer_num == 0:
                # Preventing a -1 index by mistake
                base_depth = torch.tensor(0.0, device=self.device)
            else:
                base_depth = self.cum_layer_thickness[current.layer_num - 1]

            # This is not the last entry in the list
            if i < len(self.wetting_fronts) - 1:
                next_ = self.wetting_fronts[i + 1]
                if next_.layer_num == current.layer_num:
                    sum = sum + (current.depth_cm - base_depth) * (
                        current.theta - next_.theta
                    )
                else:
                    sum = sum + (current.depth_cm - base_depth) * current.theta
            else:  # This is the last entry in the list. This must be the deepest front in the final layer
                sum += current.theta * (current.depth_cm - base_depth)

        return sum

    def initialize_starting_parameters(self, cfg: DictConfig) -> None:
        # Finish initializing variables
        # initially we start with a dry surface (no surface ponding)
        self.ponded_depth_cm = torch.tensor(0, device=self.device)
        # No. of spatial intervals used in trapezoidal integration to compute G
        self.nint = 120  # hacked, not needed to be an input option
        self.num_wetting_fronts = self.num_layers
        self.time_s = 0.0
        self.timesteps = 0.0
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
        self.precipitation_mm_per_h = torch.tensor(-1, device=self.device)
        self.PET_mm_per_h = torch.tensor(-1, device=self.device)

        # Initializing the rest of the mass balance variables to zero
        self.volprecip_cm = torch.tensor(0.0, device=self.device)
        self.volin_cm = torch.tensor(0.0, device=self.device)
        self.volend_cm = torch.tensor(0.0, device=self.device)
        self.volAET_cm = torch.tensor(0.0, device=self.device)
        self.volrech_cm = torch.tensor(0.0, device=self.device)
        self.volrunoff_cm = torch.tensor(0.0, device=self.device)
        self.volrunoff_giuh_cm = torch.tensor(0.0, device=self.device)
        self.volQ_cm = torch.tensor(0.0, device=self.device)
        self.volPET_cm = torch.tensor(0.0, device=self.device)
        self.volon_cm = torch.tensor(0.0, device=self.device)

        # setting volon and precip at the initial time to 0.0 as they determine the creation of surficail wetting front
        self.volon_timestep_cm = torch.tensor(0.0, device=self.device)
        self.precip_previous_timestep_cm = torch.tensor(0.0, device=self.device)

        # setting flux from groundwater_reservoir_to_stream to zero, will be non-zero when groundwater reservoir is added/simulated
        self.volQ_gw_cm = torch.tensor(0.0, device=self.device)

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
            next_ = i
            next_to_next = i + 1 if i + 1 < len(self.wetting_fronts) else None

            # case : wetting front passing another wetting front within a layer
            if (
                self.wetting_fronts[current].depth_cm
                > self.wetting_fronts[next_].depth_cm
                and self.wetting_fronts[current].layer_num
                == self.wetting_fronts[next_].layer_num
                and not self.wetting_fronts[next_].to_bottom
            ):
                current_mass_this_layer = self.wetting_fronts[current].depth_cm * (
                    self.wetting_fronts[current].theta
                    - self.wetting_fronts[next_].theta
                ) + self.wetting_fronts[next_].depth_cm * (
                    self.wetting_fronts[next_].theta
                    - self.wetting_fronts[next_to_next].theta
                )
                self.wetting_fronts[current].depth_cm = current_mass_this_layer / (
                    self.wetting_fronts[current].theta
                    - self.wetting_fronts[next_to_next].theta
                )

                layer_num = self.wetting_fronts[current].layer_num
                soil_num = self.layer_soil_type[layer_num]
                soil_properties = self.soils_df.iloc[soil_num]
                theta_e = torch.tensor(soil_properties["theta_e"], device=self.device)
                theta_r = torch.tensor(soil_properties["theta_r"], device=self.device)
                alpha = torch.tensor(
                    soil_properties["alpha(cm^-1)"],
                    dtype=torch.float64,
                    device=self.device,
                )
                m = torch.tensor(soil_properties["m"], device=self.device)
                n = torch.tensor(soil_properties["n"], device=self.device)
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
        for wf in self.wetting_fronts:
            wf.print()

    def wetting_fronts_cross_layer_boundary(self):
        log.debug("Layer boundary crossing...")

        for i in range(1, len(self.wetting_fronts)):
            log.debug(f"Boundary Crossing | ******* Wetting Front = {i} ******")

            current = i - 1
            next_ = i
            next_to_next = i + 1 if i + 1 < len(self.wetting_fronts) else None

            layer_num = self.wetting_fronts[current].layer_num
            soil_num = self.layer_soil_type[layer_num]
            soil_properties = self.soils_df.iloc[soil_num]
            theta_e = torch.tensor(soil_properties["theta_e"], device=self.device)
            theta_r = torch.tensor(soil_properties["theta_r"], device=self.device)
            alpha = torch.tensor(soil_properties["alpha(cm^-1)"], device=self.device)
            m = torch.tensor(soil_properties["m"], device=self.device)
            n = torch.tensor(soil_properties["n"], device=self.device)
            ksat_cm_per_h = (
                torch.tensor(soil_properties["Ks(cm/h)"])
                * self.frozen_factor[layer_num]
            )

            # TODO VERIFY THAT THIS WORKS!
            if (
                self.wetting_fronts[current].depth_cm
                > self.cum_layer_thickness[layer_num]
                and (
                    self.wetting_fronts[next_].depth_cm
                    == self.cum_layer_thickness[layer_num]
                )
                and (layer_num != (self.num_layers - 1))  # 0-based
            ):
                current_theta = min(theta_e, self.wetting_fronts[current].theta)
                overshot_depth = (
                    self.wetting_fronts[current].depth_cm
                    - self.wetting_fronts[next_].depth_cm
                )
                soil_num_next = self.layer_soil_type[layer_num + 1]
                soil_properties_next = self.soils_df.iloc[soil_num]

                se = calc_se_from_theta(
                    self.wetting_fronts[current].theta, theta_e, theta_r
                )
                self.wetting_fronts[current].psi_cm = calc_h_from_se(se, alpha, m, n)

                self.wetting_fronts[current].k_cm_per_h = calc_k_from_se(
                    se, ksat_cm_per_h, m
                )

                theta_new = calc_theta_from_h(
                    self.wetting_fronts[current].psi_cm,
                    soil_properties_next,
                    self.device,
                )

                mbal_correction = overshot_depth * (
                    current_theta - self.wetting_fronts[next_].theta
                )
                mbal_Z_correction = mbal_correction / (
                    theta_new - self.wetting_fronts[next_to_next].theta
                )

                depth_new = self.cum_layer_thickness[layer_num] + mbal_Z_correction

                self.wetting_fronts[current].depth_cm = self.cum_layer_thickness[
                    layer_num
                ]

                self.wetting_fronts[next_].theta = theta_new
                self.wetting_fronts[next_].psi_cm = self.wetting_fronts[current].psi_cm
                self.wetting_fronts[next_].depth_cm = depth_new
                self.wetting_fronts[next_].layer_num = layer_num + 1
                self.wetting_fronts[next_].dzdt_cm_per_h = self.wetting_fronts[
                    current
                ].dzdt_cm_per_h
                self.wetting_fronts[current].dzdt_cm_per_h = 0
                self.wetting_fronts[current].to_bottom = True
                self.wetting_fronts[next_].to_bottom = False

                log.debug("State after wetting fronts cross layer boundary...")
                for wf in self.wetting_fronts:
                    wf.print()

    def wetting_front_cross_domain_boundary(self):
        """
        the function lets wetting fronts of a sufficient depth interact with the lower boundary;
        called from self.lgar_move_wetting_fronts().
        :return:
        """

        bottom_flux_cm = torch.tensor(0.0, device=self.device)

        log.debug("Domain boundary crossing (bottom flux calc.)")

        for i in range(1, len(self.wetting_fronts)):
            log.debug(f"Domain boundary crossing | ***** Wetting Front = {i} ******")

            bottom_flux_cm_temp = torch.tensor(0.0, device=self.device)

            current = i - 1
            next_ = i
            next_to_next = i + 1 if i + 1 < len(self.wetting_fronts) else None

            layer_num = self.wetting_fronts[current].layer_num
            soil_num = self.layer_soil_type[layer_num]

            if (
                next_to_next is None
                and self.wetting_fronts[current].depth_cm
                > self.cum_layer_thickness[layer_num]
            ):
                bottom_flux_cm_temp = (
                    self.wetting_fronts[current].theta
                    - self.wetting_fronts[next_].theta
                ) * (
                    self.wetting_fronts[current].depth_cm
                    - self.wetting_fronts[next_].depth_cm
                )

                soil_properties = self.soils_df.iloc[soil_num]
                theta_e = torch.tensor(soil_properties["theta_e"], device=self.device)
                theta_r = torch.tensor(soil_properties["theta_r"], device=self.device)
                alpha = torch.tensor(
                    soil_properties["alpha(cm^-1)"],
                    dtype=torch.float64,
                    device=self.device,
                )
                m = torch.tensor(soil_properties["m"], device=self.device)
                n = torch.tensor(soil_properties["n"], device=self.device)
                _ksat_cm_per_h_ = (
                    soil_properties["Ks(cm/h)"] * self.frozen_factor[layer_num]
                )
                ksat_cm_per_h = torch.tensor(_ksat_cm_per_h_, device=self.device)

                self.wetting_fronts[next_].theta = self.wetting_fronts[current].theta
                se_k = calc_se_from_theta(
                    self.wetting_fronts[current].theta, theta_e, theta_r
                )
                self.wetting_fronts[next_].psi_cm = calc_h_from_se(se_k, alpha, m, n)
                self.wetting_fronts[next_].k_cm_per_h = calc_k_from_se(
                    se_k, ksat_cm_per_h, m
                )
                self.wetting_fronts.pop(i - 1)

                log.debug(
                    "State after lowest wetting front contributes to flux through the bottom boundary..."
                )
                for wf in self.wetting_fronts:
                    wf.print()

            bottom_flux_cm = bottom_flux_cm + bottom_flux_cm_temp

            log.debug(f"Bottom boundary flux = {bottom_flux_cm}")

            if bottom_flux_cm_temp != 0:
                break

        return bottom_flux_cm

    def fix_dry_over_wet_fronts(self, mass_change):
        """
        /* The function handles situation of dry over wet wetting fronts
        mainly happen when AET extracts more water from the upper wetting front
        and the front gets drier than the lower wetting front */
        :param mass_change:
        :return:
        """
        log.debug("Fix Dry over Wet Wetting Front...")

        for i in range(len(self.wetting_fronts)):
            current = i
            next_ = i + 1 if i + 1 < len(self.wetting_fronts) else None

            if next_ is not None:
                """
                // this part fixes case of upper theta less than lower theta due to AET extraction
                // also handles the case when the current and next wetting fronts have the same theta
                // and are within the same layer
                /***************************************************/
                """
                #  TODO: TEST THIS!
                if (
                    self.wetting_fronts[current].theta
                    <= self.wetting_fronts[next_].theta
                    and self.wetting_fronts[current].layer_num
                    == self.wetting_fronts[next_].layer_num
                ):
                    layer_num_k = self.wetting_fronts[current].layer_num
                    mass_before = self.calc_mass_balance()

                    self.wetting_fronts[i].pop()

                    if layer_num_k > 1:
                        soil_num_k = self.layer_soil_type[
                            self.wetting_fronts[current].layer_num
                        ]
                        soil_properties_k = self.soils_df.iloc[soil_num_k]
                        theta_e_k = torch.tensor(
                            soil_properties_k["theta_e"],
                            dtype=torch.float64,
                            device=self.device,
                        )
                        theta_r_k = torch.tensor(
                            soil_properties_k["theta_r"],
                            dtype=torch.float64,
                            device=self.device,
                        )
                        alpha_k = torch.tensor(
                            soil_properties_k["alpha(cm^-1)"],
                            dtype=torch.float64,
                            device=self.device,
                        )
                        m_k = torch.tensor(
                            soil_properties_k["m"],
                            dtype=torch.float64,
                            device=self.device,
                        )
                        n_k = torch.tensor(
                            soil_properties_k["n"],
                            dtype=torch.float64,
                            device=self.device,
                        )
                        se_k = calc_se_from_theta(
                            self.wetting_fronts[current].theta, theta_e_k, theta_r_k
                        )

                        self.wetting_fronts[current].psi_cm = calc_h_from_se(
                            se_k, alpha_k, m_k, n_k
                        )

                        while self.wetting_fronts[self.current].layer_num < layer_num_k:
                            soil_num_k1 = self.layer_soil_type[
                                self.wetting_fronts[self.current].layer_num
                            ]
                            soil_properties_k1 = self.soils_df.iloc[soil_num_k1]
                            theta_e_k = soil_properties_k1["theta_e"]
                            theta_r_k = soil_properties_k1["theta_r"]
                            vg_a_k = soil_properties_k1["alpha(cm^-1)"]
                            vg_m_k = soil_properties_k1["m"]
                            vg_n_k = soil_properties_k1["n"]
                            se_l = calc_se_from_theta(
                                self.wetting_fronts[current].theta, theta_e_k, theta_r_k
                            )
                            self.wetting_fronts[self.current].psi_cm = calc_h_from_se(
                                se_l, vg_a_k, vg_m_k, vg_n_k
                            )
                            self.wetting_fronts[self.current].theta = calc_theta_from_h(
                                self.wetting_fronts["current"].psi_cm,
                                soil_properties_k1,
                                self.device,
                            )
                            self.current = self.current + 1

                    """
                    /* note: mass_before is less when we have wetter front over drier front condition,
	                 however, lgar_calc_mass_bal returns mass_before > mass_after due to fabs(theta_current - theta_next);
	                for mass_before the functions compuates more than the actual mass; removing fabs in that function
	                might be one option, but for now we are adding fabs to mass_change to make sure we added extra water
	                back to AET after deleting the drier front */
                    """
                    mass_after = self.calc_mass_balance()
                    mass_change = mass_change + torch.abs(mass_after - mass_before)

    def create_surficial_front_func(self, ponded_depth_cm, volin, dry_depth):
        """
        // ######################################################################################
        /* This subroutine is called iff there is no surfacial front, it creates a new front and
           inserts ponded depth, and will return some amount if can't fit all water */
        // ######################################################################################
        """

        to_bottom = False
        top_front = self.wetting_fronts[0]  # Specifically pointing to the first front
        soils_data = read_soils(self, top_front)

        layer_num = 0

        delta_theta = soils_data["theta_e"] - top_front.theta

        if (
            dry_depth * delta_theta
        ) > ponded_depth_cm:  # all the ponded depth enters the soil
            volin = ponded_depth_cm
            theta_new = torch.min(
                (top_front.theta + ponded_depth_cm / dry_depth), soils_data["theta_e"]
            )
            surficial_front = WettingFront(dry_depth, theta_new, layer_num, to_bottom)
            self.wetting_fronts.insert(
                0, surficial_front
            )  # inserting the new front in the front layer
            ponded_depth_cm = torch.tensor(0.0, device=self.device)
        else:  # // not all ponded depth fits in
            volin = dry_depth * delta_theta
            ponded_depth_cm = self.ponded_depth_cm - (dry_depth * delta_theta)
            theta_new = soils_data["theta_e"]  # fmin(theta1 + (*ponded_depth_cm) /dry_depth, theta_e)
            if (
                dry_depth < self.cum_layer_thickness[0]
            ):  # checking against the first layer
                surficial_front = WettingFront(
                    dry_depth, soils_data["theta_e"], layer_num, to_bottom
                )
            else:
                surficial_front = WettingFront(
                    dry_depth, soils_data["theta_e"], layer_num, True
                )
            self.wetting_fronts.insert(0, surficial_front)

        # These calculations are allowed as we're creating a dry layer of the same soil type
        se = calc_se_from_theta(theta_new, soils_data["theta_e"], soils_data["theta_r"])
        new_front = self.wetting_fronts[0]
        new_front.psi_cm = calc_h_from_se(
            se, soils_data["alpha"], soils_data["m"], soils_data["n"]
        )

        new_front.k_cm_per_h = (
            calc_k_from_se(se, soils_data["ksat_cm_per_h"], soils_data["m"])
            * self.frozen_factor[layer_num]
        )  # // AJ - K_temp in python version for 1st layer
        new_front.dzdt_cm_per_h = torch.tensor(0.0, device=self.device)  # for now assign 0 to dzdt as it will be computed/updated in lgar_dzdt_calc function

        return ponded_depth_cm, volin

    def insert_water(self, use_closed_form_G, nint, timestep_h, precip_timestep_cm, wf_free_drainage_demand, ponded_depth_cm, volin_this_timestep):
        """
        /* The module computes the potential infiltration capacity, fp (in the lgar manuscript),
        potential infiltration capacity = the maximum amount of water that can be inserted into
        the soil depending on the availability of water.
        this module is called when a new superficial wetting front is not created
        in the current timestep, that is precipitation in the current and previous
        timesteps was greater than zero */
        :param use_closed_form_G:
        :param nint:
        :param subtimestep_h:
        :param ponded_depth_subtimestep_cm:
        :param volin_subtimestep_cm:
        :return:
        """
        wf_that_supplies_free_drainage_demand = wf_free_drainage_demand

        f_p = torch.tensor(0.0, device=self.device)
        runoff = torch.tensor(0.0, device=self.device)

        h_p = torch.clamp(ponded_depth_cm - precip_timestep_cm * timestep_h, min=0.0)  # water ponded on the surface

        current = self.wetting_fronts[0]
        current_free_drainage = self.wetting_fronts[wf_that_supplies_free_drainage_demand]
        current_free_drainage_next = self.wetting_fronts[wf_that_supplies_free_drainage_demand + 1]

        number_of_wetting_fronts = len(self.wetting_fronts)

        layer_num_fp = current_free_drainage.layer_num
        soils_data = read_soils(self, current_free_drainage)

        if number_of_wetting_fronts == self.num_layers:
            # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
            geff = torch.tensor(0.0, device=self.device)  # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
        else:
            # double theta = current_free_drainage->theta;
            theta_below = current_free_drainage_next.theta

            geff = calc_geff(use_closed_form_G, soils_data, theta_below, soils_data["theta_e"], nint, self.device)

        # if the free_drainage wetting front is the top most, then the potential infiltration capacity has the following simple form
        if layer_num_fp == 0:
            f_p = soils_data["ksat_cm_per_h"] * (1 + (geff + h_p)/current_free_drainage.depth_cm)
        else:
            # // point here to the equation in lgar paper once published
            bottom_sum = (current_free_drainage.depth_cm - self.cum_layer_thickness[layer_num_fp-1])/soils_data["ksat_cm_per_h"]

            for k in reversed(range(len(self.layer_soil_type))):
                soil_num = self.layer_soil_type[k]
                soil_properties = self.soils_df.iloc[soil_num]
                ksat_cm_per_h_k = soil_properties["ksat_cm_per_h"] * self.frozen_factor[layer_num_fp - k]

                bottom_sum = bottom_sum + (self.cum_layer_thickness[layer_num_fp - k] - self.cum_layer_thickness[layer_num_fp - (k+1)])/ soil_properties["ksat_cm_per_h"]

            f_p = (current_free_drainage.depth_cm / bottom_sum) + ((geff + h_p)*soil_properties["ksat_cm_per_h"]/(current_free_drainage.depth_cm))  #Geff + h_p

        soils_data_current = read_soils(self, self.wetting_fronts[0])  # We are using the HEAD node's data

        theta_e1 = soils_data_current["theta_e"]  # saturated theta of top layer

        # if free drainge has to be included, which currently we don't, then the following will be set to hydraulic conductivity
        # of the deeepest layer
        if (layer_num_fp == self.num_layers) and (current_free_drainage.theta == theta_e1) and (self.num_layers == number_of_wetting_fronts):
            f_p = torch.tensor(0.0, device=self.device)

        ponded_depth_temp = ponded_depth_cm

        free_drainage_demand = torch.tensor(0.0, device=self.device)

        # 'if' condition is not needed ... AJ
        if (layer_num_fp==self.num_layers) and (self.num_layers == number_of_wetting_fronts):
            ponded_depth_temp = ponded_depth_cm - f_p * timestep_h - free_drainage_demand * 0
        else:
            ponded_depth_temp = ponded_depth_cm - f_p * timestep_h - free_drainage_demand * 0

        ponded_depth_temp = torch.clamp(ponded_depth_temp, min=0.0)

        fp_cm = f_p * timestep_h + free_drainage_demand/timestep_h  # infiltration in cm

        if self.ponded_depth_max_cm > 0.0 :
            if ponded_depth_temp < self.ponded_depth_max_cm:
                runoff = torch.tensor(0.0, device=self.device)
                volin_this_timestep = torch.min(ponded_depth_cm, fp_cm)  # PTL: does this code account for the case where volin_this_timestep can not all infiltrate?
                ponded_depth_cm = ponded_depth_cm - volin_this_timestep
                return runoff, volin_this_timestep, ponded_depth_cm
            elif ponded_depth_temp > self.ponded_depth_max_cm:
              runoff = ponded_depth_temp - self.ponded_depth_max_cm
              ponded_depth_cm = self.ponded_depth_max_cm
              volin_this_timestep = fp_cm

              return runoff, volin_this_timestep, ponded_depth_cm  # TODO LOOK INTO THE POINTERS
        else:
            # if it got to this point, no ponding is allowed, either infiltrate or runoff
            # order is important here; assign zero to ponded depth once we compute volume in and runoff
            volin_this_timestep = torch.min(ponded_depth_cm, fp_cm)
            runoff = torch.tensor(0.0, device=self.device) if ponded_depth_cm < fp_cm else (ponded_depth_cm - volin_this_timestep)
            ponded_depth_cm = 0.0

        return runoff, volin_this_timestep, ponded_depth_cm

