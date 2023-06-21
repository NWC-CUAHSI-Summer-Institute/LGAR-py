import logging
import pandas as pd
import torch

log = logging.getLogger("physics.giuh")
torch.set_default_dtype(torch.float64)

def giuh_convolution_integral(runoff_m, num_giuh_ordinates, giuh_ordinates, runoff_queue_m_per_timestep):
    """
    This function solves the convolution integral involving N
    GIUH ordinates.
    :param runoff_m: (passed in as volrunoff_subtimestep_cm)
    :param num_giuh_ordinates:
    :param runoff_queue_m_per_timestep: (passed in as giuh_ordinates)
    :return:
    """
    for i in range(num_giuh_ordinates):
        runoff_queue_m_per_timestep[i] = runoff_queue_m_per_timestep[i] + (giuh_ordinates[i] * runoff_m)

    runoff_m_now = runoff_queue_m_per_timestep[0]

    # shift all the entries in preperation for the next timestep
    runoff_queue_m_per_timestep_temp = torch.zeros(num_giuh_ordinates)
    runoff_queue_m_per_timestep_temp[:-1] = runoff_queue_m_per_timestep[1:]
    runoff_queue_m_per_timestep = runoff_queue_m_per_timestep_temp
    # TODO: DO WE HAVE TO UPDATE THE POINTER HERE?

    return runoff_queue_m_per_timestep, runoff_m_now






