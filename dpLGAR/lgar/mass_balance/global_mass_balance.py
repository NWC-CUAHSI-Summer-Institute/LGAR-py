import logging

from dpLGAR.lgar.mass_balance.base_mass_balance import BaseMassBalance

log = logging.getLogger(__name__)


class GlobalMassBalance(BaseMassBalance):
    def __init__(self):
        super(GlobalMassBalance).__init__()

    def finalize_giuh_runoff(self, soil_state):
        for i in range(soil_state.num_giuh_ordinates):
            self.giuh_runoff = self.giuh_runoff + soil_state.giuh_runoff[i]

    def print(self, ending_volume):
        self.ending_volume = ending_volume
        global_error_cm = self._calc_mb()
        log.info("********************************************************* ")
        log.info("-------------------- Simulation Summary ----------------- ")
        log.info("------------------------ Mass balance ------------------- ")
        log.info(f"Initial water in soil    = {self.starting_volume.item():14f} cm")
        log.info(f"Total precipitation      = {self.precip.item():14f} cm")
        log.info(f"Total infiltration       = {self.infiltration.item():14f} cm")
        log.info(f"Final water in soil      = {self.ending_volume.item():14f} cm")
        log.info(f"Surface ponded water     = {self.ponded_water.item():14f} cm")
        log.info(f"Surface runoff           = {self.runoff.item():14f} cm")
        log.info(f"GIUH runoff              = {self.giuh_runoff.item():14f} cm")
        log.info(f"Total percolation        = {self.percolation.item():14f} cm")
        log.info(f"Total AET                = {self.AET.item():14f} cm")
        log.info(f"Total PET                = {self.PET.item():14f} cm")
        log.info(f"Total discharge (Q)      = {self.discharge.item():14f} cm")
        log.info(f"Global balance           =   {global_error_cm.item():.6e} cm")
