from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Type

from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue

from dpLGAR.plugins.neuralhydrology.utils.config import Config

log = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    cfg: DictConfig

    def __post_init__(self):
        self.class_properties = self._find_properties(Config)
        matching_cfg = self._find_matching_keys(self.cfg.models)
        self.nh_config = Config(matching_cfg)

    def _find_matching_keys(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Finding keys in an omegaconf dict that match with NH configs
        """
        matching_keys = {}
        for key, value in cfg.items():
            if key in self.class_properties:
                try:
                    matching_keys[key] = value
                except MissingMandatoryValue:
                    matching_keys[key] = None
            elif isinstance(value, DictConfig):  # Recursively search nested dictionaries
                try:
                    nested_result = self._find_matching_keys(value)
                    if nested_result:
                        matching_keys[key] = nested_result
                except MissingMandatoryValue:
                    if key in self.class_properties:
                        matching_keys[key] = None
        return matching_keys

    @staticmethod
    def _find_properties(cls: Type) -> List[str]:
        """
        Finds the properties inside of a class
        :param cls: the class you want to examine
        """
        return [k for k, v in cls.__dict__.items() if isinstance(v, property)]
