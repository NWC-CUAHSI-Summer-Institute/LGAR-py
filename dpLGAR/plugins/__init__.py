import logging
from typing import Any, Dict, List, Type

from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from dpLGAR.plugins.neuralhydrology.neuralhydrology.utils.config import Config

log = logging.getLogger(__name__)


def find_matching_keys(cfg: DictConfig, class_properties: List[str]) -> Dict[str, Any]:
    """
    Finding keys in an omegaconf dict that match with NH configs
    """
    matching_keys = {}
    for key, value in cfg.items():
        if key in class_properties:
            try:
                matching_keys[key] = value
            except MissingMandatoryValue:
                matching_keys[key] = None
        elif isinstance(value, DictConfig):  # Recursively search nested dictionaries
            try:
                nested_result = find_matching_keys(value, class_properties)
                if nested_result:
                    matching_keys[key] = nested_result
            except MissingMandatoryValue:
                if key in class_properties:
                    matching_keys[key] = None
    return matching_keys


def find_properties(cls: Type) -> List[str]:
    """
    Finds the properties inside of a class
    :param cls: the class you want to examine
    """
    return [k for k, v in cls.__dict__.items() if isinstance(v, property)]


def neural_hydrology_config_adapter(cfg: DictConfig) -> Config:
    """
    A function to map values from the config file into NH format
    :param: cfg the Hydra config file
    """
    class_properties = find_properties(Config)
    matching_cfg = find_matching_keys(cfg.models, class_properties)
    nh_config = Config(matching_cfg)
    return nh_config


class HybridConfig:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
