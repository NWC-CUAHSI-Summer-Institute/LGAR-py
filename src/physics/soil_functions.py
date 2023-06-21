"""A file to hold all soil functions"""
import logging
import pandas as pd
import torch

log = logging.getLogger("physics.soil_functions")
torch.set_default_dtype(torch.float64)


def calc_theta_from_h(
    h: torch.Tensor, soil_properties: pd.DataFrame, device: str
) -> torch.Tensor():
    """
    function to calculate theta from h

    :parameter h: the initial psi (cm)
    :parameter soil_properties: All soils and their properties
    :parameter device: the device that we're using
    :return thickness of individual layers
    """
    alpha = torch.tensor(soil_properties["alpha(cm^-1)"], device=device)
    n = torch.tensor(soil_properties["n"], device=device)
    m = torch.tensor(soil_properties["m"], device=device)
    theta_e = torch.tensor(soil_properties["theta_e"], device=device)
    theta_r = torch.tensor(soil_properties["theta_r"], device=device)
    return (
        1.0 / (torch.pow(1.0 + torch.pow(alpha * h, n), m)) * (theta_e - theta_r)
    ) + theta_r


def calc_se_from_theta(
    theta: torch.Tensor, e: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """
    function to calculate Se from theta
    :param theta: the calculated inital theta
    :param e: theta_e
    :param r: theta_r
    :return: Se this is the relative (scaled 0-1) water content, like Theta
    """
    return torch.div((theta - r), (e - r))


def calc_k_from_se(
    se: torch.Tensor, ksat: torch.Tensor, m: torch.Tensor
) -> torch.Tensor:
    """
    function to calculate K from Se
    :param se: this is the relative (scaled 0-1) water content, like Theta
    :param ksat: saturated hydraulic conductivity
    :param m: ???
    :return: hydraulic conductivity (K)
    """
    return (
        ksat
        * torch.sqrt(se)
        * torch.pow(1.0 - torch.pow(1.0 - torch.pow(se, 1.0 / m), m), 2.0)
    )


def calc_h_from_se(
    se: torch.Tensor, alpha: torch.Tensor, m: torch.Tensor, n: torch.Tensor
):
    """
    function to calculate h from Se
    """
    return 1.0 / alpha * torch.pow(torch.pow(se, (-1.0 / m)) - 1.0, (1.0 / n))


def calc_aet(
    wetting_fronts,
    PET_subtimestep_cm_per_h,
    subtimestep_h,
    wilting_point_psi_cm,
    layer_soil_type,
    soils_df,
    device,
):
    """
    /* authors : Fred Ogden and Ahmad Jan
    Translated by Tadd Bindas to Python
    year    : 2022
    the code computes actual evapotranspiration given PET.
    It uses an S-shaped function used in HYDRUS-1D (Simunek & Sejna, 2018).
    AET = PET * 1/(1 + (h/h_50) )^3
    h is the capillary head at the surface and
    h_50 is the capillary head at which AET = 0.5 * PET. */
    """
    relative_moisture_at_which_PET_equals_AET = torch.tensor(
        0.75, device=device
    )

    # Starting at the first index
    layer_num = wetting_fronts.layer_num
    soil_num = layer_soil_type[layer_num]
    soil_properties = soils_df.iloc[soil_num]
    theta_e = torch.tensor(
        soil_properties["theta_e"], device=device
    )
    theta_r = torch.tensor(
        soil_properties["theta_r"], device=device
    )
    alpha = torch.tensor(
        soil_properties["alpha(cm^-1)"], device=device
    )
    n = torch.tensor(soil_properties["m"], device=device)
    m = torch.tensor(soil_properties["n"], device=device)

    theta_fc = (theta_e - theta_r) * relative_moisture_at_which_PET_equals_AET + theta_r
    wp_head_theta = calc_theta_from_h(
        wilting_point_psi_cm, alpha, m, n, theta_e, theta_r
    )
    theta_wp = (theta_fc - wp_head_theta) * 0.5 + wp_head_theta  # theta_50 in python
    Se = calc_se_from_theta(theta_wp, theta_e, theta_r)
    psi_wp_cm = calc_h_from_se(Se, alpha, m, n)

    h_ratio = 1.0 + torch.pow((wetting_fronts.psi_cm / psi_wp_cm), 3.0)
    actual_ET_demand = PET_subtimestep_cm_per_h * (1 / h_ratio) * subtimestep_h

    if actual_ET_demand < 0:
        actual_ET_demand = torch.tensor(0.0, device=device)
    elif actual_ET_demand > (PET_subtimestep_cm_per_h * subtimestep_h):
        actual_ET_demand = PET_subtimestep_cm_per_h * subtimestep_h

    log.debug(f"AET = {actual_ET_demand:.6f}")

    return actual_ET_demand
