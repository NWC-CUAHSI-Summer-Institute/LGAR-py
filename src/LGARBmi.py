"""a file to define the LGARBMI class"""
import pandas as pd
from bmipy import Bmi
import logging
from omegaconf import OmegaConf
import time
import torch
from typing import Tuple

from src.data.read import read_forcing_data
from src.physics.giuh import giuh_convolution_integral
from src.physics.LGAR.Lgar import LGAR
from src.physics.LGAR.dry_depth import calc_dry_depth
from src.physics.LGAR.dzdt import calc_dzdt
from src.physics.LGAR.utils import calc_aet
from src.physics.WettingFront import move_wetting_fronts

log = logging.getLogger("LGARBmi")
torch.set_default_dtype(torch.float64)


class LGARBmi(Bmi):
    """The LGAR BMI class"""

    _name = "LGAR Torch"
    _input_var_names = (
        "precipitation_mm_per_h",
        "PET_mm_per_h",
        "soil_moisture_wetting_fronts",
        "soil_depth_wetting_fronts",
    )
    _output_var_names = (
        "precipitation",  # cumulative amount of precip
        "potential_evapotranspiration",  # cumulative amount of potential ET
        "actual_evapotranspiration",  # cumulative amount of actual ET
        "surface_runoff",  # direct surface runoff
        "giuh_runoff",
        "soil_storage",
        "total_discharge",
        "infiltration",
        "percolation",
        "groundwater_to_stream_recharge",
        "mass_balance",
    )

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self._model = None
        self.device = None
        self._values = {}
        self._var_units = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._grid_type = {}

        self._start_time = 0.0
        self._end_time = None
        self.timestep = None
        self.nsteps = None
        self._time_units = "s"

        # Setting the variables that track output vars
        self.precipitation = None
        self.PET_mm_per_h = None
        self.soil_moisture_fronts = None
        self.soil_thickness_fronts = None
        self.runoff = None
        self.timestep_runoff = None

    def initialize(self, config_file: str) -> None:
        """Perform startup tasks for the model.

        Perform all tasks that take place before entering the model's time
        loop, including opening files and initializing the model state. Model
        inputs are read from a text-based configuration file, specified by
        `config_file`.

        Parameters
        ----------
        config_file : str
            The path to the model configuration file.

        Notes
        -----
        Models should be refactored, if necessary, to use a
        configuration file. CSDMS does not impose any constraint on
        how configuration files are formatted, although YAML is
        recommended. A template of a model's configuration file
        with placeholder values is used by the BMI.
        """
        # Convert cfg into a DictConfig obj. We need the cfg in a string format for BMI
        self.cfg = OmegaConf.create(eval(config_file))

        self._grid_type = {0: "scalar"}
        self.device = self.cfg.device

        self._model = LGAR(self.cfg)
        self.dates, self.precipitation, self.PET = read_forcing_data(self.cfg)

        self._end_time = self._model.endtime_s.item()
        self.timestep = (
            self._model.forcing_resolution_h.item() * self.cfg.units.hr_to_sec
        )  # Converting from hours to seconds
        self.nsteps = int(self._end_time / self.timestep)

        assert self.nsteps <= int(self.precipitation.shape[0])
        if self.cfg.print_output:
            log.debug("Variables are written to file: data_variables.csv")
            log.debug("Wetting fronts state is written to file: data_layers.csv")

        self.num_giuh_ordinates = self._model.num_giuh_ordinates
        self.giuh_runoff_queue = torch.zeros(
            [self.num_giuh_ordinates], device=self.device
        )

        self.local_mass_balance = None

        self.soil_moisture_fronts = torch.zeros(
            [self._model.num_layers, self.nsteps], device=self.device
        )
        self.soil_thickness_fronts = torch.zeros(
            [self._model.num_layers, self.nsteps], device=self.device
        )
        self.runoff = torch.zeros([self.nsteps])

        self._values = {
            "precipitation_mm_per_h": self._model.precipitation_mm_per_h,
            "PET_mm_per_h": self._model.PET_mm_per_h,
            "soil_moisture_wetting_fronts": self._model.soil_moisture_wetting_fronts,
            "soil_depth_wetting_fronts": self._model.soil_depth_wetting_fronts,
        }
        self._var_units = {
            "precipitation_mm_per_h": "(mm / h)",
            "PET_mm_per_h": "(mm / h)",
            "soil_moisture_wetting_fronts": "(-)",
            "soil_depth_wetting_fronts": "m",
        }

    def update(self) -> None:
        """Advance model state by one time step.

        Perform all tasks that take place within one pass through the model's
        time loop. This typically includes incrementing all of the model's
        state variables. If the model's state variables don't change in time,
        then they can be computed by the :func:`initialize` method and this
        method can return with no action.
        """
        if self._model.sft_coupled:
            self._model.frozen_factor_hydraulic_conductivity()

        self.timestep_runoff = self.run_cycle()

    def update_until(self, time_: float) -> None:
        """Advance model state until the given time.

        Parameters
        ----------
        time : float
            A model time later than the current model time.
        """
        start_time = time.perf_counter()
        for i in range(time_):
            log.debug(f"Real Time: {self.dates[i]}")
            self.set_value("precipitation_mm_per_h", self.precipitation[i])
            self.set_value("PET_mm_per_h", self.PET[i])
            self.update()  # CALLING UPDATE
            self.set_value(
                "soil_moisture_wetting_fronts",
                self._model.soil_moisture_wetting_fronts,
            )
            self.set_value(
                "soil_depth_wetting_fronts",
                self._model.soil_depth_wetting_fronts,
            )
            # self.precipitation[i] = self.precipitation[i]
            # self.PET_mm_per_h[i] = self.PET[i]
            # self.soil_moisture_fronts[i] = self._model.soil_moisture_wetting_fronts
            # self.soil_thickness_fronts[i] = self._model.soil_depth_wetting_fronts
            # self.runoff[i] = self.timestep_runoff

        end_time = time.perf_counter()
        log.debug(f"Time Elapsed: {start_time - end_time:.6f} seconds")

    def finalize(self) -> None:
        """Perform tear-down tasks for the model.

        Perform all tasks that take place after exiting the model's time
        loop. This typically includes deallocating memory, closing files and
        printing reports.
        """
        # Saving results to csv
        if self.cfg.print_output:
            df_precip = pd.DataFrame(
                self.precipitation.detach().numpy(), columns="Precipitation (mm/hr)"
            )
            df_pet = pd.DataFrame(
                self.PET_mm_per_h.detach().numpy(), columns="Precipitation (mm/hr)"
            )
            df_moisture = pd.DataFrame(self.soil_moisture_fronts.detach().numpy())
            df_thickness = pd.DataFrame(self.soil_thickness_fronts.detach().numpy())
            df_precip.to_csv(self.cfg.save_paths.precip)
            df_pet.to_csv(self.cfg.save_paths.pet)
            df_moisture.to_csv(self.cfg.save_paths.soil_moisture)
            df_thickness.to_csv(self.cfg.save_paths.soil_thickness)

        # Authors were confused about this
        # volend_giuh_cm = torch.sum(self._model.giuh_runoff_queue_cm)

        global_error_cm = (
            self._model.volstart_cm
            + self._model.volprecip_cm
            - self._model.volrunoff_cm
            - self._model.volAET_cm
            - self._model.volon_cm
            - self._model.volrech_cm
            - self._model.volend_cm
        )

        log.info(
            f"\n---------------------- Simulation Summary  ------------------------ \n"
        )
        log.info(
            f"Time (sec)                 = {self._end_time -self._start_time:.6f} \n"
        )
        log.info(f"-------------------------- Mass balance ------------------- \n")
        log.info(f"Initial water in soil      = {self._model.volstart_cm} cm\n")
        log.info(f"Total precipitation        = {self._model.volprecip_cm}cm\n")
        log.info(f"Total Infiltration         = {self._model.volin_cm} cm\n")
        log.info(f"Final Water in Soil        = {self._model.volend_cm} cm\n")
        log.info(f"Surface ponded water       = {self._model.volon_cm} cm\n")
        log.info(f"Surface runoff             = {self._model.volrunoff_cm} cm\n")
        log.info(f"GIUH runoff                = {self._model.volrunoff_giuh} cm\n")
        log.info(f"Total percolation          = {self._model.volrech_cm} cm\n")
        log.info(f"Total AET                  = {self._model.volAET_cm} cm\n")
        log.info(f"Total PET                  = {self._model.volPET_cm} cm\n")
        log.info(f"Total discharge (Q)        = {self._model.total_Q_cm} cm\n")
        log.info(f"Global Balance             = {global_error_cm} cm\n")

    def run_cycle(self):
        """
        A function to run the model's subcycling timestep loop
        :return:
        """
        subcycles = torch.round(
            self._model.forcing_interval
        )  # ROUNDING SINCE MACHINE PRECISION SHORTENS BY 1
        num_layers = self._model.num_layers

        precip_timestep_cm = torch.tensor(0.0, device=self.device)
        PET_timestep_cm = torch.tensor(0.0, device=self.device)
        AET_timestep_cm = torch.tensor(0.0, device=self.device)
        volend_timestep_cm = self._model.calc_mass_balance()
        volin_timestep_cm = torch.tensor(0.0, device=self.device)
        volon_timestep_cm = self._model.volon_timestep_cm
        volrunoff_timestep_cm = torch.tensor(0.0, device=self.device)
        volrech_timestep_cm = torch.tensor(0.0, device=self.device)
        surface_runoff_timestep_cm = torch.tensor(0.0, device=self.device)
        volrunoff_giuh_timestep_cm = torch.tensor(0.0, device=self.device)
        volQ_timestep_cm = torch.tensor(0.0, device=self.device)
        volQ_gw_timestep_cm = torch.tensor(0.0, device=self.device)

        subtimestep_h = self._model.timestep_h
        nint = self._model.nint
        wilting_point_psi_cm = self._model.wilting_point_psi_cm
        use_closed_form_G = self._model.use_closed_form_G

        AET_thresh_Theta = torch.tensor(
            self.cfg.constants.AET_thresh_Theta, device=self.device
        )
        AET_expon = torch.tensor(self.cfg.constants.AET_expon, device=self.device)

        volend_subtimestep_cm = volend_timestep_cm
        volQ_gw_subtimestep_cm = torch.tensor(0.0, device=self.device)

        ponded_depth_max_cm = self._model.ponded_depth_max_cm

        hourly_precip_cm = (
            self.get_value_ptr("precipitation_mm_per_h") * self.cfg.units.mm_to_cm
        )  # rate [cm/hour]
        hourly_PET_cm = (
            self.get_value_ptr("PET_mm_per_h") * self.cfg.units.mm_to_cm
        )  # rate [cm/hour]
        log.debug("*** LASAM BMI Update... *** ")
        log.debug(f"Pr [cm/hr] (timestep) = {hourly_precip_cm}")
        log.debug(f"PET [cm/hr] (timestep) = {hourly_PET_cm}")

        self._model.prev_wetting_fronts = self._model.wetting_fronts.clone()

        # Ensure forcings are non-negative
        assert hourly_precip_cm >= 0.0
        assert hourly_PET_cm >= 0.0

        time_s = torch.tensor(0.0, device=self.device)
        timesteps = torch.tensor(0.0, device=self.device)
        for i in range(int(subcycles)):
            """
            /* Note unit conversion:
            Pr and PET are rates (fluxes) in mm/h
            Pr [mm/h] * 1h/3600sec = Pr [mm/3600sec]
            Model timestep (dt) = 300 sec (5 minutes for example)
            convert rate to amount
            Pr [mm/3600sec] * dt [300 sec] = Pr[mm] * 300/3600.
            in the code below, subtimestep_h is this 300/3600 factor (see initialize from config in lgar.cxx)
            """
            time_s = time_s + (subtimestep_h * self.cfg.units.hr_to_sec)
            timesteps = timesteps + 1

            log.debug(
                f"BMI Update |---------------------------------------------------------------|"
            )
            log.debug(
                f"BMI Update |Timesteps = {timesteps} Time [h] = {(time_s / 3600):.4f} Subcycle = {(i + 1):.4f} of {subcycles:.4f}"
            )

            precip_subtimestep_cm_per_h = hourly_precip_cm
            PET_subtimestep_cm_per_h = hourly_PET_cm

            ponded_depth_subtimestep_cm = (
                precip_subtimestep_cm_per_h * subtimestep_h
            )  # the amount of water on the surface before any infiltration and runoff
            ponded_depth_subtimestep_cm = (
                ponded_depth_subtimestep_cm + volin_timestep_cm
            )  # add volume of water on the surface (from the last timestep) to ponded depth as well

            precip_subtimestep_cm = (
                precip_subtimestep_cm_per_h * subtimestep_h
            )  # rate x dt = amount (portion of the water on the suface for model's timestep [cm])
            PET_subtimestep_cm = PET_subtimestep_cm_per_h * subtimestep_h
            volin_subtimestep_cm = torch.tensor(0.0, device=self.device)
            precip_previous_subtimestep_cm = self._model.precip_previous_timestep_cm

            # Calculate AET from PET if PET is non-zero
            if PET_subtimestep_cm_per_h > 0.0:
                AET_subtimestep_cm = calc_aet(
                    self._model.wetting_fronts,
                    PET_subtimestep_cm_per_h,
                    subtimestep_h,
                    wilting_point_psi_cm,
                    self._model.layer_soil_type,
                    self._model.soils_df,
                    self.device,
                )
            else:
                AET_subtimestep_cm = torch.tensor(0.0, device=self.device)

            precip_timestep_cm = precip_timestep_cm + precip_subtimestep_cm
            PET_timestep_cm = PET_timestep_cm + torch.clamp(
                PET_subtimestep_cm, min=0.0
            )  # Ensuring non-negative PET

            volstart_subtimestep_cm = self._model.calc_mass_balance()

            soil_num = self._model.layer_soil_type[
                self._model.wetting_fronts[self._model.current].layer_num
            ]
            soil_properties = self._model.soils_df.iloc[soil_num]
            theta_e = soil_properties["theta_e"]
            is_top_wf_saturated = (
                True
                if self._model.wetting_fronts[self._model.current].theta + 1e-12
                >= theta_e
                else False
            )  # PTL: sometimes a machine precision error would erroneously create a new wetting front during saturated conditions. The + 1E-12 seems to prevent this.

            # Addressed machine precision issues where volon_timestep_error could be for example -1E-17 or 1.E-20 or smaller
            # volon_timestep_cm = torch.clamp(volon_timestep_cm, min=0.0)

            # Determining if we need to create a superficial front
            create_surficial_front = (
                precip_previous_subtimestep_cm == 0.0
                and precip_subtimestep_cm > 0.0
                and volon_timestep_cm == 0
            ).item()

            # Note, we're in python so everything is 0-based
            wf_free_drainage_demand = (
                self._model.wetting_front_free_drainage()
            )  # This is assuming this function is already defined somewhere in your code

            flag = "Yes" if create_surficial_front and not is_top_wf_saturated else "No"
            log.debug(f"Create superficial wetting front? {flag}")

            if create_surficial_front and (is_top_wf_saturated is False):
                # Create a new wetting front if the following is true. Meaning there is no
                # wetting front in the top layer to accept the water, must create one.
                temp_pd = torch.tensor(0.0, device=self.device)
                move_wetting_fronts(
                    self._model,
                    subtimestep_h,
                    temp_pd,
                    wf_free_drainage_demand,
                    volend_subtimestep_cm,
                    num_layers,
                    AET_subtimestep_cm,
                )

                dry_depth = calc_dry_depth(
                    self._model, use_closed_form_G, nint, subtimestep_h
                )

                volin_subtimestep_cm = self._model.create_surficial_front_func(
                    dry_depth
                )

                self._model.prev_wetting_fronts = self._model.wetting_fronts.clone()

                volin_timestep_cm = volin_timestep_cm + volin_subtimestep_cm

                log.debug("New Wetting Front Created")
                for wf in self._model.wettting_fronts:
                    wf.print()

            if ponded_depth_subtimestep_cm > 0 and (create_surficial_front is False):
                #  infiltrate water based on the infiltration capacity given no new wetting front
                #  is created and that there is water on the surface (or raining).

                (
                    volrunoff_subtimestep_cm,
                    volin_subtimestep_cm,
                    ponded_depth_subtimestep_cm,
                ) = self._model.insert_water(
                    use_closed_form_G,
                    nint,
                    subtimestep_h,
                    precip_timestep_cm,
                    wf_free_drainage_demand,
                    ponded_depth_subtimestep_cm,
                    volin_subtimestep_cm,
                )

                volin_timestep_cm = volin_timestep_cm + volin_subtimestep_cm
                volrunoff_timestep_cm = volrunoff_timestep_cm + volrunoff_subtimestep_cm
                volrech_subtimestep_cm = volin_subtimestep_cm  # this gets updated later, probably not needed here

                volon_subtimestep_cm = ponded_depth_subtimestep_cm

                if volrunoff_subtimestep_cm < 0:
                    log.error("There is a mass balance problem")
                    raise ValueError
            else:
                if ponded_depth_subtimestep_cm < ponded_depth_max_cm:
                    # volrunoff_timestep_cm = volrunoff_timestep_cm + 0
                    volon_subtimestep_cm = ponded_depth_subtimestep_cm
                    ponded_depth_subtimestep_cm = torch.tensor(0.0, device=self.device)
                    volrunoff_subtimestep_cm = torch.tensor(0.0, device=self.device)
                else:
                    volrunoff_subtimestep_cm = (
                        ponded_depth_subtimestep_cm - ponded_depth_max_cm
                    )
                    volrunoff_timestep_cm = (
                        ponded_depth_subtimestep_cm - ponded_depth_max_cm
                    )
                    volon_subtimestep_cm = ponded_depth_max_cm
                    ponded_depth_subtimestep_cm = ponded_depth_max_cm
            if create_surficial_front is False:
                # move wetting fronts if no new wetting front is created. Otherwise, movement
                # of wetting fronts has already happened at the time of creating surficial front,
                # so no need to move them here. */
                volin_subtimestep_cm_temp = volin_subtimestep_cm

                time.sleep(0.1)  # Trying to remove segfaults
                move_wetting_fronts(
                    self._model,
                    subtimestep_h,
                    volin_subtimestep_cm,
                    wf_free_drainage_demand,
                    volend_subtimestep_cm,
                    num_layers,
                    AET_subtimestep_cm,
                )

                volrech_subtimestep_cm = volin_subtimestep_cm
                volrech_timestep_cm = volrech_timestep_cm + volrech_subtimestep_cm
                volin_subtimestep_cm = (
                    volin_subtimestep_cm_temp  # resetting the subtimestep
                )

                # / *---------------------------------------------------------------------- * /
                # // calculate derivative(dz / dt) for all wetting fronts
                calc_dzdt(
                    self._model, use_closed_form_G, nint, ponded_depth_subtimestep_cm
                )

                volend_subtimestep_cm = self._model.calc_mass_balance()
                volend_timestep_cm = volend_subtimestep_cm
                self._model.precip_previous_timestep_cm = precip_subtimestep_cm

                # /*----------------------------------------------------------------------*/
                # // mass balance at the subtimestep (local mass balance)

                local_mb = (
                    volstart_subtimestep_cm
                    + precip_subtimestep_cm
                    + volon_timestep_cm
                    - volrunoff_subtimestep_cm
                    - AET_subtimestep_cm
                    - volon_subtimestep_cm
                    - volrech_subtimestep_cm
                    - volend_subtimestep_cm
                )

                AET_timestep_cm = AET_timestep_cm + AET_subtimestep_cm
                volon_timestep_cm = volon_subtimestep_cm  # surface ponded water at the end of the timestep

                # /*----------------------------------------------------------------------*/
                # // compute giuh runoff for the subtimestep
                surface_runoff_subtimestep_cm = volrunoff_subtimestep_cm
                (
                    self.giuh_runoff_queue,
                    volrunoff_giuh_subtimestep_cm,
                ) = giuh_convolution_integral(
                    volrunoff_subtimestep_cm,
                    self._model.num_giuh_ordinates,
                    self._model.giuh_ordinates,
                    self.giuh_runoff_queue,
                )

                surface_runoff_timestep_cm = (
                    surface_runoff_timestep_cm + surface_runoff_subtimestep_cm
                )
                volrunoff_giuh_timestep_cm = (
                    volrunoff_giuh_timestep_cm + volrunoff_giuh_subtimestep_cm
                )

                # total mass of water leaving the system, at this time it is the giuh-only, but later will add groundwater component as well.
                volQ_timestep_cm = volQ_timestep_cm + volrunoff_giuh_subtimestep_cm

                # adding groundwater flux to stream channel (note: this will be updated/corrected after adding the groundwater reservoir)
                volQ_gw_timestep_cm = volQ_gw_timestep_cm + volQ_gw_subtimestep_cm
                log.debug(f"Printing wetting fronts at this subtimestep...")
                for wf in self._model.wetting_fronts:
                    wf.print()

                unexpected_error = True if torch.abs(local_mb) > 1e-6 else False
                log.debug(
                    f"\nLocal mass balance at this timestep... \n"
                    f"Error         = {local_mb.item():14.10f} \n"
                    f"Initial water = {volstart_subtimestep_cm.item():14.10f} \n"
                    f"Water added   = {precip_subtimestep_cm.item():14.10f} \n"
                    f"Ponded water  = {volon_subtimestep_cm.item():14.10f} \n"
                    f"Infiltration  = {volin_subtimestep_cm.item():14.10f} \n"
                    f"Runoff        = {volrunoff_subtimestep_cm.item():14.10f} \n"
                    f"AET           = {AET_subtimestep_cm.item():14.10f} \n"
                    f"Percolation   = {volrech_subtimestep_cm.item():14.10f} \n"
                    f"Final water   = {volend_subtimestep_cm.item():14.10f} \n"
                )

                if unexpected_error:
                    log.error(
                        f"Local mass balance (in this timestep) is {local_mb.item():14.10f}, larger than expected, needs some debugging..."
                    )
                    raise RuntimeError("Unexpected local error!")

            self._model.local_mass_balance = local_mb

            assert (
                self._model.wetting_fronts[0].depth_cm > 0.0
            )  # check on negative layer depth --> move this to somewhere else AJ (later)
            # TODO ASK ABOUT LASAM STANDALONE

        for i in range(self._model.num_wetting_fronts):
            self._model.soil_moisture_wetting_fronts[i] = self._model.wetting_fronts[
                i
            ].theta
            self._model.soil_depth_wetting_fronts[i] = (
                self._model.wetting_fronts[i].depth_cm * self.cfg.units.cm_to_m
            )
            log.debug(
                f"Wetting fronts (bmi outputs) (depth in meters, theta)= {self._model.soil_depth_wetting_fronts[i]}, {self._model.soil_moisture_wetting_fronts[i]}"
            )

        # self._model.volprecip_timestep_cm = precip_timestep_cm
        #         # self.volin_timestep_cm = volin_timestep_cm
        #         # self._model.volon_timestep_cm = volon_timestep_cm
        #         # self._model.volend_timestep_cm = volend_timestep_cm
        #         # self._model.volAET_timestep_cm = AET_timestep_cm
        #         # self._model.volrech_timestep_cm = volrech_timestep_cm
        #         # self._model.volrunoff_timestep_cm = volrunoff_timestep_cm
        #         # self._model.volQ_timestep_cm = volQ_timestep_cm
        #         # self._model.volQ_gw_timestep_cm = volQ_gw_timestep_cm
        #         # self._model.volPET_timestep_cm = PET_timestep_cm
        #         # self._model.volrunoff_giuh_timestep_cm = volrunoff_giuh_timestep_cm

        self._model.volprecip_cm = self._model.volprecip_cm + precip_timestep_cm
        self._model.volin_cm = self._model.volin_cm + volin_timestep_cm
        self._model.volon_cm = volon_timestep_cm
        self._model.volend_cm = volend_timestep_cm
        self._model.volAET_cm = self._model.volAET_cm + AET_timestep_cm
        self._model.volrech_cm = self._model.volrech_cm + volrech_timestep_cm
        self._model.volrunoff_cm = self._model.volrunoff_cm + volrunoff_timestep_cm
        self._model.volQ_cm = self._model.volQ_cm + volQ_timestep_cm
        self._model.volQ_gw_cm = self._model.volQ_gw_cm + volQ_gw_timestep_cm
        self._model.volPET_cm = self._model.volPET_cm + PET_timestep_cm
        self._model.volrunoff_giuh_cm = (
            self._model.volrunoff_giuh_cm + volrunoff_giuh_timestep_cm
        )

        # TODO: Probably Never lol XD
        #  // converted values, a struct local to the BMI and has bmi output variables
        # bmi_unit_conv.mass_balance_m        = state->lgar_mass_balance.local_mass_balance * state->units.cm_to_m;
        # bmi_unit_conv.volprecip_timestep_m  = precip_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volin_timestep_m      = volin_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volend_timestep_m     = volend_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volAET_timestep_m     = AET_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volrech_timestep_m    = volrech_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volrunoff_timestep_m  = volrunoff_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volQ_timestep_m       = volQ_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volQ_gw_timestep_m    = volQ_gw_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volPET_timestep_m     = PET_timestep_cm * state->units.cm_to_m;
        # bmi_unit_conv.volrunoff_giuh_timestep_m = volrunoff_giuh_timestep_cm * state->units.cm_to_m;

    def get_component_name(self) -> str:
        """Name of the component.

        Returns
        -------
        str
            The name of the component.
        """
        raise NotImplementedError

    def get_input_item_count(self) -> int:
        """Count of a model's input variables.

        Returns
        -------
        int
          The number of input variables.
        """
        raise NotImplementedError

    def get_output_item_count(self) -> int:
        """Count of a model's output variables.

        Returns
        -------
        int
          The number of output variables.
        """
        raise NotImplementedError

    def get_input_var_names(self) -> Tuple[str]:
        """List of a model's input variables.

        Input variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The input variables for the model.

        Notes
        -----
        Standard Names enable the CSDMS framework to determine whether
        an input variable in one model is equivalent to, or compatible
        with, an output variable in another model. This allows the
        framework to automatically connect components.

        Standard Names do not have to be used within the model.
        """
        raise NotImplementedError

    def get_output_var_names(self) -> Tuple[str]:
        """List of a model's output variables.

        Output variable names must be CSDMS Standard Names, also known
        as *long variable names*.

        Returns
        -------
        list of str
            The output variables for the model.
        """
        raise NotImplementedError

    def get_var_grid(self, name: str) -> int:
        """Get grid identifier for the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
          The grid identifier.
        """
        raise NotImplementedError

    def get_var_type(self, name: str) -> str:
        """Get data type of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The Python variable type; e.g., ``str``, ``int``, ``float``.
        """
        raise NotImplementedError

    def get_var_units(self, name: str) -> str:
        """Get units of the given variable.

        Standard unit names, in lower case, should be used, such as
        ``meters`` or ``seconds``. Standard abbreviations, like ``m`` for
        meters, are also supported. For variables with compound units,
        each unit name is separated by a single space, with exponents
        other than 1 placed immediately after the name, as in ``m s-1``
        for velocity, ``W m-2`` for an energy flux, or ``km2`` for an
        area.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The variable units.

        Notes
        -----
        CSDMS uses the `UDUNITS`_ standard from Unidata.

        .. _UDUNITS: http://www.unidata.ucar.edu/software/udunits
        """
        raise NotImplementedError

    def get_var_itemsize(self, name: str) -> int:
        """Get memory use for each array element in bytes.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            Item size in bytes.
        """
        raise NotImplementedError

    def get_var_nbytes(self, name: str) -> int:
        """Get size, in bytes, of the given variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        int
            The size of the variable, counted in bytes.
        """
        raise NotImplementedError

    def get_var_location(self, name: str) -> str:
        """Get the grid element type that the a given variable is defined on.

        The grid topology can be composed of *nodes*, *edges*, and *faces*.

        *node*
            A point that has a coordinate pair or triplet: the most
            basic element of the topology.

        *edge*
            A line or curve bounded by two *nodes*.

        *face*
            A plane or surface enclosed by a set of edges. In a 2D
            horizontal application one may consider the word “polygon”,
            but in the hierarchy of elements the word “face” is most common.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        str
            The grid location on which the variable is defined. Must be one of
            `"node"`, `"edge"`, or `"face"`.

        Notes
        -----
        CSDMS uses the `ugrid conventions`_ to define unstructured grids.

        .. _ugrid conventions: http://ugrid-conventions.github.io/ugrid-conventions
        """
        raise NotImplementedError

    def get_current_time(self) -> float:
        """Current time of the model.

        Returns
        -------
        float
            The current model time.
        """
        raise NotImplementedError

    def get_start_time(self) -> float:
        """Start time of the model.

        Model times should be of type float.

        Returns
        -------
        float
            The model start time.
        """
        raise NotImplementedError

    def get_end_time(self) -> float:
        """End time of the model.

        Returns
        -------
        float
            The maximum model time.
        """
        return self._end_time

    def get_time_units(self) -> str:
        """Time units of the model.

        Returns
        -------
        str
            The model time unit; e.g., `days` or `s`.

        Notes
        -----
        CSDMS uses the UDUNITS standard from Unidata.
        """
        raise NotImplementedError

    def get_time_step(self) -> float:
        """Current time step of the model.

        The model time step should be of type float.

        Returns
        -------
        float
            The time step used in model.
        """
        raise NotImplementedError

    def get_value(self, name: str, dest: torch.Tensor) -> torch.Tensor:
        # DONT EVER EVER EVER USE THIS!!!!!! IT BREAKS TENSOR CONNECTIVITY
        """Get a copy of values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a *copy* of a model variable, with
        the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.

        Returns
        -------
        ndarray
            The same numpy array that was passed as an input buffer.
        """
        dest = self.get_value_ptr(name).flatten()
        return dest

    def get_value_ptr(self, name: str) -> torch.Tensor:
        """Get a reference to values of the given variable.

        This is a getter for the model, used to access the model's
        current state. It returns a reference to a model variable,
        with the return type, size and rank dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.

        Returns
        -------
        array_like
            A reference to a model variable.
        """
        return self._values[name]

    def get_value_at_indices(
        self, name: str, dest: torch.Tensor, inds: torch.Tensor
    ) -> torch.Tensor:
        """Get values at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        inds : array_like
            The indices into the variable array.

        Returns
        -------
        array_like
            Value of the model variable at the given location.
        """
        return self._values[name][inds]

    def set_value(self, name: str, src: torch.Tensor) -> None:
        """Specify a new value for a model variable.

        This is the setter for the model, used to change the model's
        current state. It accepts, through *src*, a new value for a
        model variable, with the type, size and rank of *src*
        dependent on the variable.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        src : array_like
            The new value for the specified variable.
        """
        val = self.get_value_ptr(name)
        if val.dim() == 0:  # Check if tensor is a scalar
            val.data = src
        else:
            val[:] = src

    def set_value_at_indices(
        self, name: str, inds: torch.Tensor, src: torch.Tensor
    ) -> None:
        """Specify a new value for a model variable at particular indices.

        Parameters
        ----------
        name : str
            An input or output variable name, a CSDMS Standard Name.
        inds : array_like
            The indices into the variable array.
        src : array_like
            The new value for the specified variable.
        """
        val = self.get_value_ptr(name)
        val[inds] = src

    # Grid information
    def get_grid_rank(self, grid: int) -> int:
        """Get number of dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Rank of the grid.
        """
        raise NotImplementedError

    def get_grid_size(self, grid: int) -> int:
        """Get the total number of elements in the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            Size of the grid.
        """
        raise NotImplementedError

    def get_grid_type(self, grid: int) -> str:
        """Get the grid type as a string.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        str
            Type of grid as a string.
        """
        raise NotImplementedError

    # Uniform rectilinear
    def get_grid_shape(self, grid: int, shape: torch.Tensor) -> torch.Tensor:
        """Get dimensions of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        shape : ndarray of int, shape *(ndim,)*
            A numpy array into which to place the shape of the grid.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the grid's shape.
        """
        raise NotImplementedError

    def get_grid_spacing(self, grid: int, spacing: torch.Tensor) -> torch.Tensor:
        """Get distance between nodes of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        spacing : ndarray of float, shape *(ndim,)*
            A numpy array to hold the spacing between grid rows and columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's spacing.
        """
        raise NotImplementedError

    def get_grid_origin(self, grid: int, origin: torch.Tensor) -> torch.Tensor:
        """Get coordinates for the lower-left corner of the computational grid.

        Parameters
        ----------
        grid : int
            A grid identifier.
        origin : ndarray of float, shape *(ndim,)*
            A numpy array to hold the coordinates of the lower-left corner of
            the grid.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the coordinates of the grid's
            lower-left corner.
        """
        raise NotImplementedError

    # Non-uniform rectilinear, curvilinear
    def get_grid_x(self, grid: int, x: torch.Tensor) -> torch.Tensor:
        """Get coordinates of grid nodes in the x direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        x : ndarray of float, shape *(nrows,)*
            A numpy array to hold the x-coordinates of the grid node columns.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's column x-coordinates.
        """
        raise NotImplementedError

    def get_grid_y(self, grid: int, y: torch.Tensor) -> torch.Tensor:
        """Get coordinates of grid nodes in the y direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        y : ndarray of float, shape *(ncols,)*
            A numpy array to hold the y-coordinates of the grid node rows.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's row y-coordinates.
        """
        raise NotImplementedError

    def get_grid_z(self, grid: int, z: torch.Tensor) -> torch.Tensor:
        """Get coordinates of grid nodes in the z direction.

        Parameters
        ----------
        grid : int
            A grid identifier.
        z : ndarray of float, shape *(nlayers,)*
            A numpy array to hold the z-coordinates of the grid nodes layers.

        Returns
        -------
        ndarray of float
            The input numpy array that holds the grid's layer z-coordinates.
        """
        raise NotImplementedError

    def get_grid_node_count(self, grid: int) -> int:
        """Get the number of nodes in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid nodes.
        """
        raise NotImplementedError

    def get_grid_edge_count(self, grid: int) -> int:
        """Get the number of edges in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid edges.
        """
        raise NotImplementedError

    def get_grid_face_count(self, grid: int) -> int:
        """Get the number of faces in the grid.

        Parameters
        ----------
        grid : int
            A grid identifier.

        Returns
        -------
        int
            The total number of grid faces.
        """
        raise NotImplementedError

    def get_grid_edge_nodes(self, grid: int, edge_nodes: torch.Tensor) -> torch.Tensor:
        """Get the edge-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        edge_nodes : ndarray of int, shape *(2 x nnodes,)*
            A numpy array to place the edge-node connectivity. For each edge,
            connectivity is given as node at edge tail, followed by node at
            edge head.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the edge-node connectivity.
        """
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int, face_edges: torch.Tensor) -> torch.Tensor:
        """Get the face-edge connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_edges : ndarray of int
            A numpy array to place the face-edge connectivity.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-edge connectivity.
        """
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int, face_nodes: torch.Tensor) -> torch.Tensor:
        """Get the face-node connectivity.

        Parameters
        ----------
        grid : int
            A grid identifier.
        face_nodes : ndarray of int
            A numpy array to place the face-node connectivity. For each face,
            the nodes (listed in a counter-clockwise direction) that form the
            boundary of the face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the face-node connectivity.
        """
        raise NotImplementedError

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: torch.Tensor
    ) -> torch.Tensor:
        """Get the number of nodes for each face.

        Parameters
        ----------
        grid : int
            A grid identifier.
        nodes_per_face : ndarray of int, shape *(nfaces,)*
            A numpy array to place the number of nodes per face.

        Returns
        -------
        ndarray of int
            The input numpy array that holds the number of nodes per face.
        """
        raise NotImplementedError
