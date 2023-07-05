from omegaconf import DictConfig
import logging
import numpy as np
import torch
from torch import Tensor

log = logging.getLogger("models.physics.lgar.green_ampt")


def calc_giuh(global_params, runoff_m) -> Tensor:
    runoff_queue_m_per_timestep = torch.zeros([global_params.num_giuh_ordinates], device= global_params.device)
    for i in range(global_params.num_giuh_ordinates):
        runoff_queue_m_per_timestep[i] += global_params.giuh_ordinates[i] * runoff_m
    runoff_m_now = runoff_queue_m_per_timestep[0]
    for i in range(1, global_params.num_giuh_ordinates):
        runoff_queue_m_per_timestep[i - 1] = runoff_queue_m_per_timestep[i]
    runoff_queue_m_per_timestep[-1] = torch.tensor(0.0, device=global_params.device)
    return runoff_m_now
