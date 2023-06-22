"""A file to hold all water functions"""
import logging
import torch

from src.physics.LGAR.utils import (
    calc_se_from_theta,
    calc_h_from_se,
    calc_se_from_h,
    calc_k_from_se,
)
from src.physics.LGAR.utils import read_soils

log = logging.getLogger("physics.LGAR.water")
torch.set_default_dtype(torch.float64)


def insert_water(
    lgar,
    use_closed_form_G,
    nint,
    timestep_h,
    precip_timestep_cm,
    wf_free_drainage_demand,
    ponded_depth_cm,
    volin_this_timestep,
):
    """
    /* The module computes the potential infiltration capacity, fp (in the lgar manuscript),
    potential infiltration capacity = the maximum amount of water that can be inserted into
    the soil depending on the availability of water.
    this module is called when a new superficial wetting front is not created
    in the current timestep, that is precipitation in the current and previous
    timesteps was greater than zero */
    :param use_closed_form_G:
    :param nint:
    :param subtimestep_h:
    :param ponded_depth_subtimestep_cm:
    :param volin_subtimestep_cm:
    :return:
    """
    wf_that_supplies_free_drainage_demand = wf_free_drainage_demand

    f_p = torch.tensor(0.0, device=lgar.device)
    runoff = torch.tensor(0.0, device=lgar.device)

    h_p = torch.clamp(
        ponded_depth_cm - precip_timestep_cm * timestep_h, min=0.0
    )  # water ponded on the surface

    current = lgar.wetting_fronts[0]
    current_free_drainage = lgar.wetting_fronts[wf_that_supplies_free_drainage_demand]
    current_free_drainage_next = lgar.wetting_fronts[
        wf_that_supplies_free_drainage_demand + 1
    ]

    number_of_wetting_fronts = len(lgar.wetting_fronts)

    layer_num_fp = current_free_drainage.layer_num
    soils_data = read_soils(lgar, current_free_drainage)

    if number_of_wetting_fronts == lgar.num_layers:
        # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
        geff = torch.tensor(
            0.0, device=lgar.device
        )  # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
    else:
        # double theta = current_free_drainage->theta;
        theta_below = current_free_drainage_next.theta

        geff = calc_geff(
            use_closed_form_G,
            soils_data,
            theta_below,
            soils_data["theta_e"],
            nint,
            lgar.device,
        )

    # if the free_drainage wetting front is the top most, then the potential infiltration capacity has the following simple form
    if layer_num_fp == 0:
        f_p = soils_data["ksat_cm_per_h"] * (
            1 + (geff + h_p) / current_free_drainage.depth_cm
        )
    else:
        # // point here to the equation in lgar paper once published
        bottom_sum = (
            current_free_drainage.depth_cm - lgar.cum_layer_thickness[layer_num_fp - 1]
        ) / soils_data["ksat_cm_per_h"]

        for k in reversed(range(len(lgar.layer_soil_type))):
            soil_num = lgar.layer_soil_type[k]
            soil_properties = lgar.soils_df.iloc[soil_num]
            ksat_cm_per_h_k = (
                soil_properties["ksat_cm_per_h"] * lgar.frozen_factor[layer_num_fp - k]
            )

            bottom_sum = (
                bottom_sum
                + (
                    lgar.cum_layer_thickness[layer_num_fp - k]
                    - lgar.cum_layer_thickness[layer_num_fp - (k + 1)]
                )
                / soil_properties["ksat_cm_per_h"]
            )

        f_p = (current_free_drainage.depth_cm / bottom_sum) + (
            (geff + h_p)
            * soil_properties["ksat_cm_per_h"]
            / (current_free_drainage.depth_cm)
        )  # Geff + h_p

    soils_data_current = read_soils(
        lgar, lgar.wetting_fronts[0]
    )  # We are using the HEAD node's data

    theta_e1 = soils_data_current["theta_e"]  # saturated theta of top layer

    # if free drainge has to be included, which currently we don't, then the following will be set to hydraulic conductivity
    # of the deeepest layer
    if (
        (layer_num_fp == lgar.num_layers)
        and (current_free_drainage.theta == theta_e1)
        and (lgar.num_layers == number_of_wetting_fronts)
    ):
        f_p = torch.tensor(0.0, device=lgar.device)

    ponded_depth_temp = ponded_depth_cm

    free_drainage_demand = torch.tensor(0.0, device=lgar.device)

    # 'if' condition is not needed ... AJ
    if (layer_num_fp == lgar.num_layers) and (
        lgar.num_layers == number_of_wetting_fronts
    ):
        ponded_depth_temp = (
            ponded_depth_cm - f_p * timestep_h - free_drainage_demand * 0
        )
    else:
        ponded_depth_temp = (
            ponded_depth_cm - f_p * timestep_h - free_drainage_demand * 0
        )

    ponded_depth_temp = torch.clamp(ponded_depth_temp, min=0.0)

    fp_cm = f_p * timestep_h + free_drainage_demand / timestep_h  # infiltration in cm

    if lgar.ponded_depth_max_cm > 0.0:
        if ponded_depth_temp < lgar.ponded_depth_max_cm:
            runoff = torch.tensor(0.0, device=lgar.device)
            volin_this_timestep = torch.min(
                ponded_depth_cm, fp_cm
            )  # PTL: does this code account for the case where volin_this_timestep can not all infiltrate?
            ponded_depth_cm = ponded_depth_cm - volin_this_timestep
            return runoff, volin_this_timestep, ponded_depth_cm
        elif ponded_depth_temp > lgar.ponded_depth_max_cm:
            runoff = ponded_depth_temp - lgar.ponded_depth_max_cm
            ponded_depth_cm = lgar.ponded_depth_max_cm
            volin_this_timestep = fp_cm

            return (
                runoff,
                volin_this_timestep,
                ponded_depth_cm,
            )  # TODO LOOK INTO THE POINTERS
    else:
        # if it got to this point, no ponding is allowed, either infiltrate or runoff
        # order is important here; assign zero to ponded depth once we compute volume in and runoff
        volin_this_timestep = torch.min(ponded_depth_cm, fp_cm)
        runoff = (
            torch.tensor(0.0, device=lgar.device)
            if ponded_depth_cm < fp_cm
            else (ponded_depth_cm - volin_this_timestep)
        )
        ponded_depth_cm = 0.0

    return runoff, volin_this_timestep, ponded_depth_cm
