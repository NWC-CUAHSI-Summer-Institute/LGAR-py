from omegaconf import DictConfig
import logging

log = logging.getLogger("sanity_checks")


def soil_types(cfg: DictConfig) -> bool:
    if cfg.variables.num_soil_types > (cfg.tests.max_soil_types - 1):
        log.error("Too many soil types specified.  Increase MAX_NUM_SOIL_TYPES in data/config")
        return True
    if cfg.variables.num_soil_layers > (cfg.tests.max_soil_types - 1):
        log.error("Too many soil types specified.  Increase MAX_NUM_SOIL_LAYERS in code.")
        return True
    return False


class DataError(Exception):
    """A custom exception for demonstration purposes."""
    pass