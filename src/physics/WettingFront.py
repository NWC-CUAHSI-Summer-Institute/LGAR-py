"""a file to hold code concerning a wetting front"""
import logging
import torch

from src.physics.soil_functions import calc_se_from_theta, calc_h_from_se, calc_k_from_se, calc_theta_from_h

log = logging.getLogger("physics.WettingFront")
torch.set_default_dtype(torch.float64)


class WettingFront:
    def __init__(self, depth, theta, layer_num, bottom_flag):
        super().__init__()  # Use front_num as val for the base Node class
        self.depth_cm = depth
        self.theta = theta
        self.layer_num = layer_num
        self.to_bottom = bottom_flag
        self.dzdt_cm_per_h = 0.0
        self.psi_cm = None
        self.k_cm_per_h = None

    def print(self):
        log.debug(
            f"******** Layer {self.layer_num} ********\n"
            f"(depth_cm: {self.depth_cm:.6f})\n"
            f"(theta: {self.theta:.6f})\n"
            f"(to_bottom: {self.to_bottom})\n"
            f"(dzdt_cm_per_h: {self.dzdt_cm_per_h:.6f})\n"
            f"(K_cm_per_h: {self.k_cm_per_h:.6f})\n"
            f"(psi_cm: {self.psi_cm:.6f})\n"
        )


def move_wetting_fronts(
    lgar,
    timestep_h,
    volin_cm,
    wf_free_drainage_demand,
    old_mass,
    num_layers,
    AET_demand_cm,
):
    """
    the function moves wetting fronts, merge wetting fronts and does the mass balance correction when needed
    @param current : wetting front pointing to the current node of the current state
    @param next    : wetting front pointing to the next node of the current state
    @param previous    : wetting front pointing to the previous node of the current state
    @param current_old : wetting front pointing to the current node of the previous state
    @param next_old    : wetting front pointing to the next node of the previous state
    @param head : pointer to the first wetting front in the list of the current state

    Note: '_old' denotes the wetting_front or variables at the previous timestep (or state)
    :param lgar: the LGAR obj
    :param subtimestep_h:
    :param volin_subtimestep_cm:
    :param wf_free_drainage_demand:
    :param volend_subtimestep_cm:
    :param num_layers:
    :param AET_subtimestep_cm:
    :return:
    """
    column_depth = lgar.cum_layer_thickness[-1]
    previous = lgar.current
    current = lgar.current

    precip_mass_to_add = volin_cm  # water to be added to the soil
    bottom_boundary_flux_cm = torch.tensor(
        0.0, device=lgar.device
    )  # water leaving the system through the bottom boundary
    volin_cm = torch.tensor(
        0.0, device=lgar.device
    )  # assuming that all the water can fit in, if not then re-assign the left over water at the end

    number_of_wetting_fronts = len(lgar.wetting_fronts) - 1  # Indexed at 0
    last_wetting_front_index = number_of_wetting_fronts
    for i in reversed(range(len(lgar.wetting_fronts))):
        """
        /* ************************************************************ */
        // main loop advancing all wetting fronts and doing the mass balance
        // loop goes over deepest to top most wetting front
        // wf denotes wetting front
        """
        if i == 0 and number_of_wetting_fronts > 0:
            current = i
            next_ = i + 1
            previous = None

            current_old = i
            next_old = i + 1
        elif i < number_of_wetting_fronts:
            current = i
            next_ = i + 1
            previous = i - 1

            current_old = i
            next_old = i + 1
        elif i == number_of_wetting_fronts:
            current = i
            next_ = None
            previous = i - 1

            current_old = i
            next_old = None
        layer_num = lgar.wetting_fronts[current].layer_num
        soil_num = lgar.layer_soil_type[layer_num]
        soil_properties = lgar.soils_df.iloc[soil_num]
        theta_e = torch.tensor(soil_properties["theta_e"], device=lgar.device)
        theta_r = torch.tensor(soil_properties["theta_r"], device=lgar.device)
        alpha = torch.tensor(soil_properties["alpha(cm^-1)"], device=lgar.device)
        m = torch.tensor(soil_properties["m"], device=lgar.device)
        n = torch.tensor(soil_properties["n"], device=lgar.device)

        layer_num_above = (
            layer_num if i == 0 else lgar.wetting_fronts[previous].layer_num
        )
        layer_num_below = (
            layer_num + 1
            if i == last_wetting_front_index
            else lgar.wetting_fronts[next_].layer_num
        )

        log.debug(
            f"Layers (current, above, below) == {layer_num} {layer_num_above} {layer_num_below} \n"
        )
        free_drainage_demand = torch.tensor(0.0, device=lgar.device)
        actual_ET_demand = AET_demand_cm

        if i < last_wetting_front_index and layer_num_below != layer_num:
            deepest_wetting_front(lgar, current, next_, soil_properties, layer_num, layer_num_below)
        if (
            i == number_of_wetting_fronts
            and layer_num_below != layer_num
            and number_of_wetting_fronts == (num_layers - 1)
        ):
            """
            // case to check if the number of wetting fronts are equal to the number of layers, i.e., one wetting front per layer
              /*************************************************************************************/
              /* For example, 3 layers and 3 wetting fronts in a state. psi profile is constant, and theta profile is non-uniform due
              to different van Genuchten parameters
                theta profile       psi profile  (constant head)
               _____________       ______________
                         |                   |
               __________|__       __________|___
                   |                     |
               ________|____       __________|___
                   |                         |
               ____|________       __________|___
            */
            """
            log.debug(
                f"case (number_of_wetting_fronts equal to num_layers) : l {i} == num_layers (0-base) {num_layers - 1} == num_wetting_fronts{number_of_wetting_fronts}"
            )

            # this is probably not needed, as dz / dt = 0 for the deepest wetting front
            lgar.wetting_fronts[current].depth_cm = lgar.wetting_fronts[
                current
            ].depth_cm + (lgar.wetting_fronts[current].dzdt_cm_per_h * timestep_h)

            delta_thetas = torch.zeros([num_layers], device=lgar.device)
            delta_thickness = torch.zeros([num_layers], device=lgar.device)
            psi_cm_old = lgar.wetting_fronts[current_old].psi_cm

            psi_cm = lgar.wetting_fronts[current].psi_cm

            # mass = delta(depth) * delta(theta)
            prior_mass = (
                lgar.wetting_fronts[current_old].depth_cm
                - lgar.cum_layer_thickness[layer_num - 1]
            ) * (
                lgar.wetting_fronts[current_old].theta - 0.0
            )  # 0.0 = next_old->theta
            new_mass = (
                lgar.wetting_fronts[current].depth_cm
                - lgar.cum_layer_thickness[layer_num - 1]
            ) * (
                lgar.wetting_fronts[current].theta - 0.0
            )  # 0.0 = next->theta;

            for j in range(0, number_of_wetting_fronts):
                soil_num_ = lgar.layer_soil_type[j]
                soil_properties_ = lgar.soils_df.iloc[soil_num_]

                # using psi_cm_old for all layers because the psi is constant across layers in this particular case
                theta_old = calc_theta_from_h(psi_cm_old, soil_properties_, lgar.device)
                theta_below_old = torch.tensor(0.0, device=lgar.device)
                local_delta_theta_old = theta_old - theta_below_old
                if j == 0:
                    layer_thickness = lgar.cum_layer_thickness[j]
                else:
                    layer_thickness = (
                        lgar.cum_layer_thickness[j] - lgar.cum_layer_thickness[j - 1]
                    )

                prior_mass = prior_mass + layer_thickness * local_delta_theta_old

                theta = calc_theta_from_h(psi_cm, soil_properties_, lgar.device)
                theta_below = 0.0

                new_mass = new_mass + layer_thickness * (theta - theta_below)
                delta_thetas[j] = theta_below
                delta_thickness[j] = layer_thickness

            delta_thickness[layer_num] = (
                lgar.wetting_fronts[current].depth_cm
                - lgar.cum_layer_thickness[layer_num - 1]
            )
            free_drainage_demand = 0

            if wf_free_drainage_demand == i:
                prior_mass = (
                    prior_mass
                    + precip_mass_to_add
                    - (free_drainage_demand + actual_ET_demand)
                )

            # theta mass balance computes new theta that conserves the mass; new theta is assigned to the current wetting front
            theta_new = lgar.theta_mass_balance(
                layer_num,
                soil_num,
                psi_cm,
                new_mass,
                prior_mass,
                delta_thetas,
                delta_thickness,
                soil_properties,
            )

            lgar.wetting_fronts[current].theta = torch.minimum(theta_new, theta_e)
            se = calc_se_from_theta(
                lgar.wetting_fronts[current].theta, theta_e, theta_r
            )
            lgar.wetting_fronts[current].psi_cm = calc_h_from_se(se, alpha, m, n)
        if i < last_wetting_front_index and layer_num == layer_num_below:
            raise NotImplementedError
        if i == 0:
            """
            // if f_p (predicted infiltration) causes theta > theta_e, mass correction is needed.
            // depth of the wetting front is increased to close the mass balance when theta > theta_e.
            // l == 0 is the last iteration (top most wetting front), so do a check on the mass balance)
            // this part should be moved out of here to a subroutine; add a call to that subroutine
            """
            soil_num_k1 = lgar.layer_soil_type[wf_free_drainage_demand]
            theta_e_k1 = lgar.soils_df.iloc[soil_num_k1]["theta_e"]

            wf_free_drainage = lgar.wetting_fronts[wf_free_drainage_demand]
            mass_timestep = (old_mass + precip_mass_to_add) - (
                actual_ET_demand + free_drainage_demand
            )

            # Making sure that the mass is correct
            assert old_mass > 0.0

            if torch.abs(wf_free_drainage.theta - theta_e_k1) < 1e-15:
                current_mass = lgar.calc_mass_balance()

                mass_balance_error = torch.abs(
                    current_mass - mass_timestep
                )  # mass error

                factor = torch.tensor(1.0, device=lgar.device)
                switched = False
                tolerance = 1e-12

                # check if the difference is less than the tolerance
                if mass_balance_error <= tolerance:
                    pass
                    # return current_mass

                depth_new = wf_free_drainage.depth_cm

                # loop to adjust the depth for mass balance
                while torch.abs(mass_balance_error - tolerance) > 1e-12:
                    if current_mass < mass_timestep:
                        depth_new = (
                            depth_new + torch.tensor(0.01, device=lgar.device) * factor
                        )
                        switched = False
                    else:
                        if not switched:
                            switched = True
                            factor = factor * torch.tensor(0.001, device=lgar.device)
                        depth_new = depth_new - (
                            torch.tensor(0.01, device=lgar.device) * factor
                        )

                    wf_free_drainage.depth_cm = depth_new

                    current_mass = lgar.calc_mass_balance()
                    mass_balance_error = torch.abs(current_mass - mass_timestep)

    """
      // **************************** MERGING WETTING FRONT ********************************

      /* In the python version of LGAR, wetting front merging, layer boundary crossing, and lower boundary crossing
         all occur in a loop that happens after wetting fronts have been moved. This prevents the model from crashing,
         as there are rare but possible cases where multiple merging / layer boundary crossing events will happen in
         the same time step. For example, if two wetting fronts cross a layer boundary in the same time step, it will
         be necessary for merging to occur before layer boundary crossing. So, LGAR-C now approaches merging in the
         same way as in LGAR-Py, where wetting fronts are moved, then a new loop does merging for all wetting fronts,
         then a new loop does layer boundary corssing for all wetting fronts, then a new loop does merging again for
         all wetting fronts, and then a new loop does lower boundary crossing for all wetting fronts. this is a thorough
         way to deal with these scenarios. */
    """

    lgar.merge_wetting_fronts()  # Merge

    lgar.wetting_fronts_cross_layer_boundary()  # Cross

    lgar.merge_wetting_fronts()  # Merge

    bottom_boundary_flux_cm = (
        bottom_boundary_flux_cm + lgar.wetting_front_cross_domain_boundary()
    )

    volin_cm = bottom_boundary_flux_cm
    mass_change = torch.tensor(0.0, device=lgar.device)
    lgar.fix_dry_over_wet_fronts(mass_change)
    log.debug(f"mass change/adjustment (dry_over_wet case) = {mass_change}")

    if torch.abs(mass_change) > 1.0e-7:
        AET_demand_cm = AET_demand_cm - mass_change

    """
    /***********************************************/
    // make sure all psi values are updated
    """

    for front in lgar.wetting_fronts:
        soil_num_k = lgar.layer_soil_type[front.layer_num]
        soil_properties_k = lgar.soils_df.iloc[soil_num_k]
        theta_e_k = torch.tensor(soil_properties_k["theta_e"], device=lgar.device)
        theta_r_k = soil_properties_k["theta_r"]
        alpha_k = soil_properties_k["alpha(cm^-1)"]
        m_k = soil_properties_k["m"]
        n_k = soil_properties_k["n"]

        ksat_cm_per_h_k = (
            lgar.frozen_factor[front.layer_num] * soil_properties_k["Ks(cm/h)"]
        )

        se = calc_se_from_theta(front.theta, theta_e_k, theta_r_k)
        front.psi_cm = calc_h_from_se(se, alpha_k, m_k, n_k)
        front.K_cm_per_h = calc_k_from_se(se, ksat_cm_per_h_k, m_k)

    log.debug(f"Moving/merging wetting fronts done...")

    # Just a check to make sure that, when there is only 1 layer, then the existing wetting front is at the correct depth.
    if len(lgar.wetting_fronts) == 1:
        if lgar.wetting_fronts[0].depth_cm != lgar.cum_layer_thickness[0]:
            lgar.wetting_fronts[0].depth_cm = lgar.cum_layer_thickness[0]


def deepest_wetting_front(lgar, current, next_, soil_properties, layer_num, layer_num_below):
    """// case to check if the wetting front is at the interface, i.e. deepest wetting front within a layer
    // psi of the layer below is already known/updated, so we that psi to compute the theta of the deepest current layer
    // this condition can be replace by current->to_depth = FALSE && l<last_wetting_front_index
    /*             _____________
       layer_above             |
                            ___|
               |
                   ________|____    <----- wetting fronts at the interface have same psi value
                   |
       layer current    ___|
                       |
                   ____|________
       layer_below     |
                  _____|________
    */
    /*************************************************************************************/
    """
    next_psi_cm = lgar.wetting_fronts[next_].psi_cm
    log.debug(
        f"case (deepest wetting front within layer) : layer_num {layer_num} != layer_num_below {layer_num_below}"
    )
    lgar.wetting_fronts[current].theta = calc_theta_from_h(
        next_psi_cm, soil_properties, lgar.device
    )
    lgar.wetting_fronts[current].psi_cm = next_psi_cm