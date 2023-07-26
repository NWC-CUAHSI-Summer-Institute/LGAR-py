"""A file to contain all LGAR modules"""
import logging

import torch

from dpLGAR.lgar.giuh import calc_giuh

log = logging.getLogger(__name__)


def run(
    model,
    precip,
    pet,
    subtimestep_h,
):
    """
    The main LGAR function for simulating runoff in soils
    """
    model.top_layer.copy_states()
    model.subtimestep_mb.precip = precip * subtimestep_h
    model.subtimestep_mb.PET = torch.clamp(pet * subtimestep_h, min=0.0)
    model.subtimestep_mb.previous_precip = model.global_mb.previous_precip.clone()
    model.subtimestep_mb.ponded_depth = (
        model.subtimestep_mb.precip + model.local_mb.ponded_water
    )
    model.subtimestep_mb.ponded_water = torch.tensor(0.0)
    model.subtimestep_mb.percolation = torch.tensor(0.0)
    model.subtimestep_mb.runoff = torch.tensor(0.0)
    model.subtimestep_mb.inflitration = torch.tensor(0.0)
    model.subtimestep_mb.AET = torch.tensor(0.0)
    # Determining wetting cases
    create_surficial_front = _create_surficial_front(
        model, model.subtimestep_mb.previous_precip, model.subtimestep_mb.precip
    )
    wf_free_drainage_demand = calc_wetting_front_free_drainage(model)
    model.top_layer.set_wf_free_drainage_demand(wf_free_drainage_demand)
    is_top_layer_saturated = model.top_layer.is_saturated()
    if pet > 0.0:
        model.subtimestep_mb.AET = model.top_layer.calc_aet(pet, subtimestep_h)
    model.local_mb.precip = model.local_mb.precip + model.subtimestep_mb.precip
    model.local_mb.PET = model.local_mb.PET + torch.max(
        model.subtimestep_mb.PET, torch.tensor(0.0)
    )
    model.subtimestep_mb.starting_volume = model.top_layer.mass_balance()
    if create_surficial_front and is_top_layer_saturated is False:
        # ------------------------------------------------------------------------------------------------------
        # /* create a new wetting front if the following is true. Meaning there is no
        #    wetting front in the top layer to accept the water, must create one. */
        temp_pd = torch.tensor(0.0)
        # // move the wetting fronts without adding any water; this is done to close the mass balance
        temp_pd, model.subtimestep_mb.AET = move_wetting_front(
            temp_pd,
            model.subtimestep_mb.AET,
            model.subtimestep_mb.ending_volume,
            subtimestep_h,
            model.local_mb.bottom_boundary_flux,
        )
        # depth of the surficial front to be created
        dry_depth = model.top_layer.calc_dry_depth(subtimestep_h)
        (
            model.subtimestep_mb.ponded_depth,
            model.subtimestep_mb.inflitration,
        ) = model.top_layer.create_surficial_front(
            dry_depth,
            model.subtimestep_mb.ponded_depth,
            model.subtimestep_mb.inflitration,
        )
        model.top_layer.copy_states()
        model.local_mb.infiltration = (
            model.local_mb.infiltration + model.subtimestep_mb.infiltration
        )
    if create_surficial_front is False and model.subtimestep_mb.ponded_depth > 0:
        # ------------------------------------------------------------------------------------------------------
        # /* infiltrate water based on the infiltration capacity given no new wetting front
        #    is created and that there is water on the surface (or raining). */
        (
            model.subtimestep_mb.runoff,
            model.subtimestep_mb.inflitration,
            model.subtimestep_mb.ponded_depth,
        ) = model.top_layer.insert_water(
            subtimestep_h,
            model.subtimestep_mb.precip,
            model.subtimestep_mb.ponded_depth,
            model.subtimestep_mb.inflitration,
        )

        model.local_mb.infiltration = (
            model.local_mb.infiltration + model.subtimestep_mb.inflitration
        )
        model.local_mb.runoff = model.local_mb.runoff + model.subtimestep_mb.runoff
        model.subtimestep_mb.percolation = (
            model.subtimestep_mb.inflitration
        )  # this gets updated later, probably not needed here

        model.subtimestep_mb.ponded_water = model.subtimestep_mb.ponded_depth
    else:
        update_ponded_depth(model)
    if create_surficial_front is False:
        # ------------------------------------------------------------------------------------------------------
        # /* move wetting fronts if no new wetting front is created. Otherwise, movement
        #    of wetting fronts has already happened at the time of creating surficial front,
        #    so no need to move them here. */
        infiltration_temp = model.subtimestep_mb.inflitration.clone()
        (
            model.subtimestep_mb.inflitration,
            model.subtimestep_mb.AET,
        ) = move_wetting_front(
            model.subtimestep_mb.inflitration,
            model.subtimestep_mb.AET,
            model.subtimestep_mb.ending_volume,
            subtimestep_h,
            model.local_mb.bottom_boundary_flux,
        )
        model.subtimestep_mb.percolation = model.subtimestep_mb.inflitration.clone()
        model.local_mb.percolation = (
            model.local_mb.percolation + model.subtimestep_mb.percolation
        )  # Make sure the values aren't getting lost
        model.subtimestep_mb.inflitration = infiltration_temp

    # ----------------------------------------------------------------------------------------------------------
    # calculate derivative (dz/dt) for all wetting fronts
    model.top_layer.calc_dzdt(model.subtimestep_mb.ponded_depth)
    model.subtimestep_mb.ending_volume = model.top_layer.mass_balance()
    previous_precip = model.subtimestep_mb.precip
    ending_volume = model.subtimestep_mb.ending_volume

    # ----------------------------------------------------------------------------------------------------------
    # mass balance at the subtimestep (local mass balance)
    # model.subtimestep_mb.print()
    model.local_mb.AET = model.local_mb.AET + model.subtimestep_mb.AET
    ponded_water = model.subtimestep_mb.ponded_water

    # ----------------------------------------------------------------------------------------------------------
    # compute giuh runoff for the subtimestep if there is runoff or runoff recently
    if (
        model.subtimestep_mb.giuh_runoff_queue.sum() > 0
        or model.subtimestep_mb.runoff > 0
    ):
        (
            model.subtimestep_mb.giuh_runoff,
            model.subtimestep_mb.giuh_runoff_queue,
        ) = calc_giuh(
            model.soil_state,
            model.subtimestep_mb.giuh_runoff_queue,
            model.subtimestep_mb.runoff,
        )
        model.local_mb.giuh_runoff = (
            model.local_mb.giuh_runoff + model.subtimestep_mb.giuh_runoff
        )
        model.local_mb.discharge = (
            model.local_mb.discharge + model.subtimestep_mb.giuh_runoff
        )
        model.local_mb.groundwater_discharge = model.local_mb.groundwater_discharge


def _create_surficial_front(model, previous_precip, precip):
    """
    Checks the volume of water on the surface, and if it's raining or has recently rained
    to determine if any water has infiltrated the surface

    :param previous_precip:
    :return:
    """
    # This enusures we don't add extra mass from a previous storm event
    has_no_previous_precip = (previous_precip == 0.0).item()
    is_it_raining = (precip > 0.0).item()
    is_there_ponded_water = (model.subtimestep_mb.ponded_water == 0).item()
    return has_no_previous_precip and is_it_raining and is_there_ponded_water


def calc_wetting_front_free_drainage(model):
    """
    A function to determine the bottom-most layer impacted by infiltration
    :return:
    """
    # Starting at 0 since python is 0-based
    wf_that_supplies_free_drainage_demand = model.top_layer.wetting_fronts[0]
    return model.top_layer.calc_wetting_front_free_drainage(
        wf_that_supplies_free_drainage_demand.psi_cm,
        wf_that_supplies_free_drainage_demand,
    )


def move_wetting_front(
    model,
    infiltration,
    AET_sub,
    ending_volume_sub,
    subtimestep_h,
    bottom_boundary_flux,
):
    num_wetting_fronts = model.top_layer.calc_num_wetting_fronts()
    infiltration = model.bottom_layer.move_wetting_fronts(
        infiltration,
        AET_sub,
        ending_volume_sub,
        num_wetting_fronts,
        subtimestep_h,
    )
    model.top_layer.merge_wetting_fronts()
    model.top_layer.wetting_fronts_cross_layer_boundary()
    model.top_layer.merge_wetting_fronts()
    bottom_boundary_flux = (
        bottom_boundary_flux + model.top_layer.wetting_front_cross_domain_boundary()
    )
    infiltration = bottom_boundary_flux
    mass_change = model.top_layer.fix_dry_over_wet_fronts()
    if torch.abs(mass_change) > 1e-7:
        AET_sub = AET_sub - mass_change
    model.top_layer.update_psi()
    return infiltration, AET_sub


def update_ponded_depth(model):
    if model.subtimestep_mb.ponded_depth < model.soil_state.ponded_depth_max:
        model.subtimestep_mb.runoff = torch.tensor(0.0)
        model.local_mb.runoff = model.local_mb.runoff + model.subtimestep_mb.runoff
        model.subtimestep_mb.ponded_water = model.subtimestep_mb.ponded_depth
        model.subtimestep_mb.ponded_depth = torch.tensor(0.0)

    else:
        # This is the timestep that adds runoff
        model.subtimestep_mb.runoff = (
            model.subtimestep_mb.ponded_depth - model.soil_state.ponded_depth_max
        )
        model.local_mb.runoff = model.local_mb.runoff + model.subtimestep_mb.runoff
        model.subtimestep_mb.ponded_water = model.subtimestep_mb.ponded_depth
        model.subtimestep_mb.ponded_depth = model.soil_state.ponded_depth_max