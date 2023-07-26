import logging

from dpLGAR.lgar.mass_balance.base_mass_balance import BaseMassBalance

log = logging.getLogger(__name__)


class LocalMassBalance(BaseMassBalance):
    def __init__(self):
        super(LocalMassBalance, self).__init__()

    def print(self, top_layer):
        top_layer.print()
        local_mb = self._calc_mb()
        log.info(f"Local mass balance at timestep {i}:")
        log.info(f"Error         = {local_mb.item():14.10f}")
        log.info(f"Initial water = {self.starting_volume.item():14.10f}")
        log.info(f"Water added   = {self.precip.item():14.10f}")
        log.info(f"Ponded water  = {self.ponded_water.item():14.10f}")
        log.info(f"Infiltration  = {self.infiltration.item():14.10f}")
        log.info(f"Runoff        = {self.runoff.item():14.10f}")
        log.info(f"AET           = {self.AET.item():14.10f}")
        log.info(f"Percolation   = {self.percolation.item():14.10f}")
        log.info(f"Final water   = {self.ending_volume.item():14.10f}")