import logging
import torch
from torch import Tensor

log = logging.getLogger("modelzoo.physics.lgar.giuh")


def calc_giuh(global_params, giuh_runoff_queue, runoff) -> (Tensor, Tensor):
    """
    Calculates GIUH runoff based on the GIUH parameters given
    :param global_params:
    :param giuh_runoff_queue:
    :param runoff:
    :return:
    """
    giuh_runoff_queue = giuh_runoff_queue + (global_params.giuh_ordinates * runoff)
    runoff_now = giuh_runoff_queue[0]
    giuh_runoff_queue = torch.roll(giuh_runoff_queue, shifts=-1)
    giuh_runoff_queue[-1] = torch.tensor(0.0, device=global_params.device)
    return runoff_now, giuh_runoff_queue
