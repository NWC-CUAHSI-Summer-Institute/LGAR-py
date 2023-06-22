"""A file to hold all soil functions"""
import logging
import torch

log = logging.getLogger("physics.LGAR.dzdt")
torch.set_default_dtype(torch.float64)


def calc_dzdt(lgar, use_closed_form_G, nint, ponded_depth_subtimestep_cm):
    """
    code to calculate velocity of fronts
    equations with full description are provided in the lgar paper (currently under review) */
    :param use_closed_form_G:
    :param nint:
    :param ponded_depth_subtimestep_cm:
    :return:
    """
    log.debug(f"Calculating dz/dt")

    dzdt = torch.tensor(0.0, device=lgar.device)
    for i in range(len(lgar.wetting_fronts) - 1):
        """
        we're done calculating dZ/dt's when at the end of the list (hence len -1)
        """
        current = lgar.wetting_fronts[i]
        layer_num = current.layer_num  # what layer the front is in
        k_cm_per_h = current.k_cm_per_h  # K(theta)

        if k_cm_per_h <= 0:
            log.debug(f"K is zero: layer:{layer_num} {k_cm_per_h}")
            raise ValueError("K is zero")

        depth_cm = current.depth_cm
        soil_num = lgar.layer_soil_type[layer_num]
        soil_properties = lgar.soils_df.iloc[soil_num]
        theta_e = torch.tensor(soil_properties["theta_e"], device=lgar.device)
        theta_r = torch.tensor(soil_properties["theta_r"], device=lgar.device)
        alpha = torch.tensor(soil_properties["alpha(cm^-1)"], device=lgar.device)
        m = torch.tensor(soil_properties["m"], device=lgar.device)
        n = torch.tensor(soil_properties["n"], device=lgar.device)
        h_min_cm = torch.tensor(soil_properties["h_min_cm"], device=lgar.device)
        ksat_cm_per_h = (
            torch.tensor(soil_properties["Ks(cm/h)"]) * lgar.frozen_factor[layer_num]
        )
        bc_lambda = torch.tensor(soil_properties["bc_lambda"], device=lgar.device)
        bc_psib_cm = torch.tensor(soil_properties["bc_psib_cm"], device=lgar.device)

        next = lgar.wetting_fronts[i + 1]
        theta_1 = next.theta
        theta_2 = current.theta

        # needed for multi-layered dz/dt equation.  Equal to sum from n=1 to N-1 of (L_n/K_n(theta_n))
        bottom_sum = torch.tensor(0.0, device=lgar.device)

        if current.to_bottom:
            current.dzdt_cm_per_h = torch.tensor(0.0, device=lgar.device)
        elif layer_num > 1:
            bottom_sum = (
                bottom_sum
                + (depth_cm - lgar.cum_layer_thickness[layer_num - 1]) / k_cm_per_h
            )
        else:
            if theta_1 > theta_2:
                log.error("theta_1 cannot be larger than theta_2")
                raise ValueError

            # TODO GEOFF IMPLEMENTATION
            raise NotImplementedError
