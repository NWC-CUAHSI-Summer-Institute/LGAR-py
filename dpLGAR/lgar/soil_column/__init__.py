# import logging
#
# from omegaconf import DictConfig
# import torch
#
# from dpLGAR.lgar.soil_column.base_state import BaseState
# from dpLGAR.lgar.soil_column.c_soil_state import CSoilState
#
#
# log = logging.getLogger(__name__)
#
#
# def get_soil_state(cfg: DictConfig) -> BaseState:
#     """Creates the global soil params that we're using within LGAR
#
#     Parameters
#     ----------
#     mass_balance : str
#         The mass balance you want to set up
#
#     Returns
#     -------
#
#     Raises
#     ------
#     NotImplementedError
#         If there is an issue creating the soil layers from the given data
#     """
#     if cfg.datzoo.soil_state.lower() == "c_soil":
#         SoilState = CSoilState
#     else:
#         raise NotImplementedError(f"No Soil State Implemented")
#
#     ss = SoilState(cfg=cfg, ponded_depth=ponded_depth)
#     return ss
#