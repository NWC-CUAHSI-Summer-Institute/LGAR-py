from eto import ETo
from functools import cached_property
import logging
import math
import os

from omegaconf import DictConfig
import torch.nn as nn

log = logging.getLogger(__name__)


class FAO_PET(nn.Module):
    """
    # References:
    # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html#eq:1
    # https://www.fao.org/3/x0490e/x0490e07.htm
    # https://github.com/NOAA-OWP/evapotranspiration/blob/1e971ffe784ade3c7ab322ccce6f49f48e942e3d/src/pet.c
    """
    def __init__(self, cfg: DictConfig) -> None:
        super(FAO_PET, self).__init__()
        self.cfg = cfg

    def calc_PET(self):
        PET = ETo(self.input_forcing, freq='H', lon=self.lon, TZ_lon=self.lon, z_msl=self.elevation,
                  lat=self.lat).eto_fao()
        PET = PET.fillna(0)
        PET = PET / self.cfg.conversions.m_to_mm / self.cfg.conversions.hr_to_sec  # mm/hr to m/hr  to m/s

        return PET

    def forward(self, nldas_forcing):


    def prepare_input(self, df):
        # Convert the time column to DateTime type
        df['time'] = pd.to_datetime(df['date'])
        df['date'] = pd.to_datetime(df['date'])

        # Set the time column as the index
        df.set_index('time', inplace=True)

        # Calculate relative humidity for each row
        df["RH_mean"] = df.apply(self.calculate_relative_humidity, axis=1)

        # Actual vapor pressure
        df["e_a"] = df.apply(self.calculate_actual_vapor_pressure_Pa, axis=1) / self.cfg.conversions.to_kilo

        # Mean find speed
        df['U_z'] = (df["wind_u"] + df["wind_v"]) / 2

        # Unit conversion
        df['R_s'] = df["shortwave_radiation"] * 0.0036  # self.cfg.conversions.day_to_sec / self.cfg.conversions.to_mega
        df['P'] = df["pressure"] / self.cfg.conversions.to_kilo
        df['T_mean'] = df["temperature"]

        input_forcing = df[["date", "R_s", "P", "T_mean", "e_a", "RH_mean", "U_z"]]

        # {
        #     'date': df.index,
        #     'R_s': df["shortwave_radiation"] * self.cfg.conversions.day_to_sec / self.cfg.conversions.to_mega,     # (W/m2) -> (MJ/m2/day)
        #     'P': df["pressure"] / self.cfg.conversions.to_kilo, # (Pa) -> (kPa)
        #     'T_mean': df["temperature"], # (deg C) -> (deg C)
        #     'e_a': df["Actual_Vapor_Pressure"] / self.cfg.conversions.to_kilo,  # (Pa) -> kPa?
        #     'RH_mean': df["Relative_Humidity"], # (-)
        #     'U_z': df["mean_wind_speed"], # (m/s) -> (m/s)
        # }

        return input_forcing

    def calculate_relative_humidity(self, row):
        air_sat_vap_press_Pa = self.calc_air_saturation_vapor_pressure_Pa(row)
        actual_vapor_pressure_Pa = self.calculate_actual_vapor_pressure_Pa(row)
        relative_humidity = actual_vapor_pressure_Pa / air_sat_vap_press_Pa
        return relative_humidity

    @staticmethod
    def calc_air_saturation_vapor_pressure_Pa(row):
        air_temperature_C = row["temperature"]
        air_sat_vap_press_Pa = 611.0 * math.exp(17.27 * air_temperature_C / (237.3 + air_temperature_C))
        return air_sat_vap_press_Pa

    @staticmethod
    def calculate_actual_vapor_pressure_Pa(row):
        # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html#eq:1
        q = row["specific_humidity"]  # Specific humidity
        p = row["pressure"]  # Atmospheric pressure in pascals
        actual_vapor_pressure_Pa = q * p / (0.622 + 0.378 * q)
        return actual_vapor_pressure_Pa