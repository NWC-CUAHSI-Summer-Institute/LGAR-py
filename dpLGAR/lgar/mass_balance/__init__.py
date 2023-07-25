# import logging
#
# from omegaconf import DictConfig
# import torch
#
# from dpLGAR.lgar.mass_balance.base_mass_balance import BaseMassBalance
# from dpLGAR.lgar.mass_balance.global_mass_balance import GlobalMassBalance
# from dpLGAR.lgar.mass_balance.local_mass_balance import LocalMassBalance
#
# log = logging.getLogger(__name__)
#
#
# def get_mass_balance(mass_balance: str, cfg: DictConfig) -> BaseMassBalance:
#     """Creates the global mass balance that we're using within LGAR
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
#     if mass_balance.lower() == "local":
#         MassBalance = LocalMassBalance
#     elif mass_balance.lower() == "global":
#         MassBalance = GlobalMassBalance
#     else:
#         raise NotImplementedError(f"No Mass Balance Implemented")
#
#     mb = MassBalance(cfg=cfg)
#     return mb
