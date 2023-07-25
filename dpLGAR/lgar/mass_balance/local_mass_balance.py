import logging

from dpLGAR.lgar.mass_balance.base_mass_balance import BaseMassBalance

log = logging.getLogger(__name__)


class LocalMassBalance(BaseMassBalance):
    def __init__(self):
        super(LocalMassBalance).__init__()

    def print(self, top_layer):
        self.top_layer.print()
        log.info(f"Local mass balance at timestep {i}:")
        log.info(f"Error         = {local_mb.item():14.10f}")
        log.info(f"Initial water = {starting_volume_sub.item():14.10f}")
        log.info(f"Water added   = {precip_sub.item():14.10f}")
        log.info(f"Ponded water  = {ponded_water_sub.item():14.10f}")
        log.info(f"Infiltration  = {infiltration_sub.item():14.10f}")
        log.info(f"Runoff        = {runoff_sub.item():14.10f}")
        log.info(f"AET           = {AET_sub.item():14.10f}")
        log.info(f"Percolation   = {percolation_sub.item():14.10f}")
        log.info(f"Final water   = {ending_volume_sub.item():14.10f}")