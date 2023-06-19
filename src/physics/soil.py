"""A file to hold all soil functions"""

import torch


def calc_theta_from_h(h: torch.Tensor, alpha: torch.Tensor, m: torch.Tensor, n: torch.Tensor, theta_e: torch.Tensor, theta_r: torch.Tensor) -> torch.Tensor():
    """
    function to calculate theta from h

    :argument h the initial psi (cm)
    :argument alpha
    :argument m
    :argument n
    :argument theta_e
    :argument theta_r
    """
    return (1.0 / (torch.pow(1.0 + torch.pow(alpha * h, n), m)) * (theta_e - theta_r)) + theta_r