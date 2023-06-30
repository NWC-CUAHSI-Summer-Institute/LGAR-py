import logging
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.models.physics.layers.WettingFront import WettingFront
from lgartorch.models.physics.lgar.aet import calc_aet
from lgartorch.models.physics.lgar.green_ampt import calc_geff
from lgartorch.models.physics.utils import (
    calc_theta_from_h,
    calc_se_from_theta,
    calc_h_from_se,
    calc_k_from_se,
)

log = logging.getLogger("models.physics.layers.Layer")


class Layer:
    def __init__(
        self,
        global_params,
        layer_index: int,
        c: Tensor,
        alpha: torch.nn.Parameter,
        n: torch.nn.Parameter,
        ksat: torch.nn.Parameter,
        texture_map: dict,
        previous_layer=None,
    ):
        """
        A layer of soil (within the soil stack).
        Each soil layer can have many wetting fronts and several properties
        :param cfg: The DictConfig
        :param global_params: many of the values within the cfg file, but as tensors
        :param c: All soil attributes
        :param alpha: All alpha van Genuchten params
        :param n: All n van Genuchten params
        :param ksat: All saturated hydraulic conductivity params
        :param is_top: TBD if this is necessary. Rn it's always true
        """
        super().__init__()
        self.global_params = global_params
        self.num_layers = global_params.num_layers
        self.layer_num = layer_index
        self.layer_thickness = self.global_params.layer_thickness_cm[self.layer_num]
        self.cumulative_layer_thickness = self.global_params.cum_layer_thickness[
            self.layer_num
        ]
        self.soil_type = self.global_params.layer_soil_type[self.layer_num]
        self.texture = texture_map[self.soil_type]
        self.attributes = c[self.soil_type]
        self.alpha_layer = alpha[self.soil_type]
        self.n_layer = n[self.soil_type]
        self.ksat_layer = ksat[self.soil_type]

        # For mass balance
        self.tolerance = torch.tensor(1e-12, device=self.global_params.device)

        # Setting up wetting fronts. Each layer can have many wetting fronts
        self.wetting_fronts = []
        self.wetting_fronts.append(
            WettingFront(
                self.global_params,
                self.cumulative_layer_thickness,
                self.layer_num,
                self.attributes,
                self.ksat_layer,
            )
        )
        self.previous_fronts = None
        self.wf_free_drainage_demand = None
        self.previous_layer = previous_layer
        self.next_layer = None
        if (
            layer_index < global_params.num_layers - 1
        ):  # Checking to see if there is a layer below this one
            self.next_layer = Layer(
                global_params,
                layer_index + 1,
                c,
                alpha,
                n,
                ksat,
                texture_map,
                previous_layer=self,
            )
        self.previous_state = self.deepcopy()

    def deepcopy(self):
        """
        Creating a persisted copy of the previous wetting fronts and states
        :return:
        """
        # TODO WORK ON DEEP COPY!!!!!!! THIS IS A PROBLEM
        state = {}
        state["wetting_fronts"] = []
        for i in range(len(self.wetting_fronts)):
            wf = WettingFront(
                            self.global_params,
                            self.cumulative_layer_thickness,
                            self.layer_num,
                            self.attributes,
                            self.ksat_layer,
                        )
            state["wetting_fronts"].append(self.wetting_fronts[i].deepcopy(wf))
        return state

    def copy_states(self):
        self.previous_state = self.deepcopy()
        if self.next_layer is not None:
            self.next_layer.copy_states()
        else:
            return None

    def calc_wetting_front_free_drainage(
        self, psi, wf_that_supplies_free_drainage_demand
    ):
        """
         /*
         finds the wetting front that corresponds to psi (head) value closest to zero
         (i.e., saturation in terms of psi). This is the wetting front that experiences infiltration
         and actual ET based on precipitatona and PET, respectively. For example, the actual ET
         is extracted from this wetting front plus the wetting fronts above it.
         Note: the free_drainage name came from its python version, which is probably not the correct name.
         */
        :return:
        """
        for i in range(len(self.wetting_fronts)):
            current_front = self.wetting_fronts[i]
            if current_front.psi_cm <= psi:
                psi = current_front.psi_cm
                wf_that_supplies_free_drainage_demand = current_front
            else:
                # Checking for machine precision errors
                if torch.isclose(current_front.psi_cm, psi, atol=1e-8):
                    psi = current_front.psi_cm
                    wf_that_supplies_free_drainage_demand = current_front
        if self.next_layer is not None:
            return self.next_layer.calc_wetting_front_free_drainage(
                psi, wf_that_supplies_free_drainage_demand
            )
        else:
            return wf_that_supplies_free_drainage_demand

    def set_wf_free_drainage_demand(self, wf_free_drainage_demand):
        self.wf_free_drainage_demand = wf_free_drainage_demand
        if self.next_layer is not None:
            self.next_layer.set_wf_free_drainage_demand(wf_free_drainage_demand)
        else:
            return None

    def calc_num_wetting_fronts(self) -> int:
        if self.next_layer is not None:
            return len(self.wetting_fronts) + self.next_layer.calc_num_wetting_fronts()
        else:
            return len(self.wetting_fronts)

    def populate_delta_thickness(
        self, psi_cm_old, psi_cm, prior_mass, new_mass, delta_thetas, delta_thickness
    ):
        """
        Populating the delta theta and delta thickness between layers
        :param delta_theta:
        :param delta_thickness:
        :return:
        """
        m = self.attributes[self.global_params.soil_index["m"]]
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        theta_old = calc_theta_from_h(
            psi_cm_old, self.alpha_layer, self.n_layer, m, theta_e, theta_r
        )
        theta_below_old = torch.tensor(0.0, device=self.global_params.device)
        local_delta_theta_old = theta_old - theta_below_old
        layer_thickness = self.global_params.layer_thickness_cm[self.layer_num]
        prior_mass = prior_mass + layer_thickness * local_delta_theta_old
        theta = calc_theta_from_h(
            psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
        )
        theta_below = 0.0

        new_mass = new_mass + layer_thickness * (theta - theta_below)
        delta_thetas[self.layer_num] = theta_below
        delta_thickness[self.layer_num] = layer_thickness
        if self.next_layer.layer_num < (self.num_layers - 1):
            return self.next_layer.populate_delta_thickness(
                psi_cm_old, psi_cm, prior_mass, new_mass, delta_thetas, delta_thickness
            )
        else:
            return new_mass, prior_mass, delta_thetas, delta_thickness

    def recalculate_mass(self, psi_cm, new_mass, delta_thetas, delta_thickness):
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        m = self.attributes[self.global_params.soil_index["m"]]
        theta_layer = calc_theta_from_h(
            psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
        )
        new_mass = new_mass + delta_thickness[self.layer_num] * (
            theta_layer - delta_thetas[self.layer_num]
        )
        if self.next_layer.layer_num < (self.num_layers - 1):
            return self.next_layer.recalculate_mass(
                psi_cm, new_mass, delta_thetas, delta_thickness
            )
        else:
            return new_mass

    def theta_mass_balance(
        self,
        psi_cm,
        new_mass,
        prior_mass,
        delta_thetas,
        delta_thickness,
    ):
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        m = self.attributes[self.global_params.soil_index["m"]]
        delta_mass = torch.abs(new_mass - prior_mass)

        # flag that determines capillary head to be incremented or decremented
        switched = False
        factor = torch.tensor(1.0, device=self.global_params.device)
        theta = torch.tensor(
            0.0, device=self.global_params.device
        )  # this will be updated and returned
        psi_cm_loc_prev = psi_cm
        delta_mass_prev = delta_mass
        count_no_mass_change = torch.tensor(0.0, device=self.global_params.device)
        break_no_mass_change = torch.tensor(5.0, device=self.global_params.device)

        # check if the difference is less than the tolerance
        if delta_mass <= self.tolerance:
            theta = calc_theta_from_h(
                psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
            )
            return theta

        # the loop increments/decrements the capillary head until mass difference between
        # the new and prior is within the tolerance
        while delta_mass > self.tolerance:
            if new_mass > prior_mass:
                psi_cm = psi_cm + 0.1 * factor
                switched = False
            else:
                if not switched:
                    switched = True
                    factor = factor * 0.1

                psi_cm_loc_prev = psi_cm
                psi_cm = psi_cm - 0.1 * factor

                if psi_cm < 0 and psi_cm_loc_prev != 0:
                    psi_cm = psi_cm_loc_prev * 0.1

            theta = calc_theta_from_h(
                psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
            )
            new_mass = delta_thickness[self.layer_num] * (
                theta - delta_thetas[self.layer_num]
            )

            new_mass = self.find_front_layer().recalculate_mass(
                psi_cm, new_mass, delta_thetas, delta_thickness
            )
            delta_mass = torch.abs(new_mass - prior_mass)

            # stop the loop if the error between the current and previous psi is less than 10^-15
            # 1. enough accuracy, 2. the algorithm can't improve the error further,
            # 3. avoid infinite loop, 4. handles a corner case when prior mass is tiny (e.g., <1.E-5)
            # printf("A1 = %.20f, %.18f %.18f %.18f %.18f \n ",fabs(psi_cm_loc - psi_cm_loc_prev) , psi_cm_loc, psi_cm_loc_prev, factor, delta_mass);
            if torch.abs(psi_cm - psi_cm_loc_prev) < 1e-15 and factor < 1e-13:
                break
            if torch.abs(delta_mass - delta_mass_prev) < 1e-15:
                count_no_mass_change += 1
            else:
                count_no_mass_change = 0
            if count_no_mass_change == break_no_mass_change:
                break
            if psi_cm <= 0 and psi_cm_loc_prev < 1e-50:
                break
            delta_mass_prev = delta_mass
        return theta

    def base_case(self, percolation, aet, subtimestep, neighboring_fronts):
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
        :return:
        """
        # this is probably not needed, as dz / dt = 0 for the deepest wetting front
        current_front = neighboring_fronts["current_front"]
        previous_current_front = neighboring_fronts["previous_current_front"]

        current_front.depth = current_front.depth + current_front.dzdt * subtimestep
        delta_thetas = torch.zeros([self.num_layers], device=self.global_params.device)
        delta_thickness = torch.zeros(
            [self.num_layers], device=self.global_params.device
        )
        psi_cm_old = previous_current_front.psi_cm

        psi_cm = current_front.psi_cm
        try:
            base_thickness = self.previous_layer.cumulative_layer_thickness
        except AttributeError:
            base_thickness = torch.tensor(0.0, device=self.global_params.device)
        # mass = delta(depth) * delta(theta)
        # 0.0 = next->theta;
        prior_mass = (previous_current_front.depth - base_thickness) * (
            previous_current_front.theta - 0.0
        )
        new_mass = (current_front.depth - base_thickness) * (current_front.theta - 0.0)

        (
            new_mass,
            prior_mass,
            delta_thetas,
            delta_thickness,
        ) = self.find_front_layer().populate_delta_thickness(
            psi_cm_old, psi_cm, prior_mass, new_mass, delta_thetas, delta_thickness
        )
        delta_thickness[self.layer_num] = current_front.depth - base_thickness

        free_drainage_demand = 0
        if self.wf_free_drainage_demand.layer_num == self.layer_num:
            prior_mass = prior_mass + percolation - (free_drainage_demand + aet)

        # theta mass balance computes new theta that conserves the mass; new theta is assigned to the current wetting front
        theta_new = self.theta_mass_balance(
            psi_cm,
            new_mass,
            prior_mass,
            delta_thetas,
            delta_thickness,
        )
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        m = self.attributes[self.global_params.soil_index["m"]]
        current_front.theta = torch.minimum(theta_new, theta_e)
        se = calc_se_from_theta(current_front.theta, theta_e, theta_r)
        current_front.psi_cm = calc_h_from_se(se, self.alpha_layer, m, self.n_layer)

    def deepest_layer_front(self, neighboring_fronts):
        """
         // case to check if the wetting front is at the interface, i.e. deepest wetting front within a layer
        // psi of the layer below is already known/updated, so we that psi to compute the theta of the deepest current layer
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
        :param neighboring_fronts: A dictionary containing pointers to all the current wetting front
        and its neighboring fronts
        :return:
        """
        current_front = neighboring_fronts["current_front"]
        next_front = neighboring_fronts["next_front"]
        m = self.attributes[self.global_params.soil_index["m"]]
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        current_front.theta = calc_theta_from_h(
            next_front.psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
        )
        current_front.psi_cm = next_front.psi_cm

    def wetting_front_in_layer(self):
        """
        // case to check if the 'current' wetting front is within the layer and not at the layer's interface
        // layer_num == layer_num_below means there is another wetting front below the current wetting front
        // and they both belong to the same layer (in simple words, wetting fronts not at the interface)
        // l < last_wetting_front_index means that the current wetting front is not the deepest wetting front in the domain
        /*************************************************************************************/
        :return:
        """
        pass

    def check_column_mass(self, free_drainage_demand, old_mass, percolation, aet):
        """
        // if f_p (predicted infiltration) causes theta > theta_e, mass correction is needed.
        // depth of the wetting front is increased to close the mass balance when theta > theta_e.
        // l == 1 is the last iteration (top most wetting front), so do a check on the mass balance)
        // this part should be moved out of here to a subroutine; add a call to that subroutine
            :return:
        """
        theta_e_k1 = self.wf_free_drainage_demand.theta_e

        # Making sure that the mass is correct
        # assert (old_mass > 0.0).item()

        mass_timestep = (old_mass + percolation) - (aet + free_drainage_demand)

        if torch.abs(self.wf_free_drainage_demand.theta - theta_e_k1) < self.tolerance:
            # Correct Mass
            current_mass = self.mass_balance()
            mass_balance_error = torch.abs(current_mass - mass_timestep)
            switched = False
            factor = torch.tensor(1.0, device=self.global_params.device)
            depth_new = self.wf_free_drainage_demand.depth
            # loop to adjust the depth for mass balance
            while torch.abs(mass_balance_error - self.tolerance) > 1e-12:
                if current_mass < mass_timestep:
                    depth_new = (
                        depth_new
                        + torch.tensor(0.01, device=self.global_params.device) * factor
                    )
                    switched = False
                else:
                    if not switched:
                        switched = True
                        factor = factor * torch.tensor(
                            0.001, device=self.global_params.device
                        )
                    depth_new = depth_new - (
                        torch.tensor(0.01, device=self.global_params.device) * factor
                    )

                self.wf_free_drainage_demand.depth_cm = depth_new

                current_mass = self.mass_balance()
                mass_balance_error = torch.abs(current_mass - mass_timestep)

    def get_neighboring_fronts(self, i):
        """
        Gets the current, previous, and next fronts for the current model state
        Also gets the previous state's current and next fronts
        :param i:
        :return:
        """
        neighboring_fronts = {}
        neighboring_fronts["current_front"] = self.wetting_fronts[i]
        neighboring_fronts["previous_current_front"] = self.previous_state[
            "wetting_fronts"
        ][i]
        neighboring_fronts["next_front"] = None
        neighboring_fronts["previous_front"] = None
        neighboring_fronts["previous_next_front"] = None
        if i > 0:
            neighboring_fronts["previous_front"] = self.wetting_fronts[i - 1]
        if i < (len(self.wetting_fronts) - 1):
            neighboring_fronts["next_front"] = self.wetting_fronts[i + 1]
            neighboring_fronts["previous_next_front"] = self.previous_state[
                "wetting_fronts"
            ][i + 1]
        if self.next_layer is not None:
            if i == (len(self.wetting_fronts) - 1):
                neighboring_fronts["next_front"] = self.next_layer.wetting_fronts[0]
                neighboring_fronts["previous_next_front"] = self.previous_state[
                    "wetting_fronts"
                ][0]
        if self.previous_layer is not None:
            if i == 0:
                neighboring_fronts[
                    "previous_front"
                ] = self.previous_layer.wetting_fronts[-1]
        return neighboring_fronts

    def get_extended_neighbors(self, i):
        neighboring_fronts = self.get_neighboring_fronts(i)
        neighboring_fronts["next_to_next_front"] = None
        if i < (len(self.wetting_fronts) - 2):
            neighboring_fronts["next_to_next_front"] = self.wetting_fronts[i + 2]
        else:
            if neighboring_fronts["next_front"] is not None:
                if (
                    neighboring_fronts["next_front"].layer_num
                    != neighboring_fronts["current_front"].layer_num
                ):
                    if len(self.next_layer.wetting_fronts) > 1:
                        neighboring_fronts[
                            "next_to_next_front"
                        ] = self.next_layer.wetting_fronts[1]
                    else:
                        if self.next_layer.next_layer is not None:
                            neighboring_fronts[
                                "next_to_next_front"
                            ] = self.next_layer.next_layer.wetting_fronts[0]
                else:
                    neighboring_fronts[
                        "next_to_next_front"
                    ] = self.next_layer.wetting_fronts[0]
        return neighboring_fronts

    def calc_aet(self, pet: Tensor, subtimestep_h: float) -> Tensor:
        """
        ONLY CALLED FROM TOP LAYER
        Calculates the Actual Evapotranspiration for each layer
        :param pet: Potential evapotranspiration
        :param subcycle_length_h: the length of each subcycle step (in hours)
        :return:
        """
        top_wetting_front = self.wetting_fronts[0]
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_index["theta_r"]]
        m = self.attributes[self.global_params.soil_index["m"]]
        aet = calc_aet(
            self.global_params,
            subtimestep_h,
            pet,
            top_wetting_front.psi_cm,
            theta_e,
            theta_r,
            m,
            self.alpha_layer,
            self.n_layer,
        )
        return aet

    def is_saturated(self):
        """
        Determining if the top layer's first wetting front is saturated
        If saturated, then there will be runoff
        :return:
        """
        top_wetting_front = self.wetting_fronts[0]
        theta_e = self.attributes[self.global_params.soil_index["theta_e"]]
        return True if top_wetting_front.theta >= theta_e else False

    def mass_balance(self) -> Tensor:
        """
        A function that calculates the mass inside of the current layer
        If `next_layer` is not None, then we iterate through the soil stack to
        find the mass underneath
        :return:
        """
        sum = torch.tensor(0, dtype=torch.float64)
        if self.layer_num == 0:
            base_depth = torch.tensor(0.0, device=self.global_params.device)
        else:
            # The base depth is the depth at the top of the layer
            base_depth = self.cumulative_layer_thickness - self.layer_thickness
        if len(self.wetting_fronts) > 1:
            # Iterate through the list elements except the last one
            for i, wf in enumerate(self.wetting_fronts[:-1]):
                current_front = self.wetting_fronts[i]
                next_front = self.wetting_fronts[i + 1]
                sum = sum + (current_front.depth - base_depth) * (
                    current_front.theta - next_front.theta
                )
            last_front = self.wetting_fronts[-1]
            sum = sum + (last_front.depth - base_depth) * last_front.theta
        else:
            current_front = self.wetting_fronts[0]
            sum = sum + (current_front.depth - base_depth) * current_front.theta
        if self.next_layer is not None:
            return sum + self.next_layer.mass_balance()
        else:
            return sum

    def is_passing(self, current_front, next_front):
        """
        # case : wetting front passing another wetting front within a layer
        :param current_front: the current wetting front
        :param next_front: the next wetting front
        :return:
        """
        larger_depth = current_front.depth > next_front.depth
        same_layer = current_front.layer_num == next_front.layer_num
        is_passing = larger_depth and same_layer and not next_front.to_bottom
        return is_passing

    def merge_wetting_fronts(self):
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
          :return:
        """
        layer_fronts = self.get_len_layers()
        for i in range(layer_fronts):
            extended_neighbors = self.get_extended_neighbors(i)
            is_passing = self.is_passing(
                extended_neighbors["current_front"], extended_neighbors["next_front"]
            )
            if is_passing:
                self.pass_front(extended_neighbors)
        if self.next_layer is not None:
            self.next_layer.merge_wetting_fronts()
        else:
            # We're on the last layer. We can't merge this to anything
            return None

    def pass_front(self, extended_neighbors):
        current_front = extended_neighbors["current_front"]
        next_front = extended_neighbors["current_front"]
        next_to_next_front = extended_neighbors["current_front"]
        current_mass_this_layer = current_front.depth_cm * (
            current_front.theta - next_front.theta
        ) + next_front.depth_cm * (next_front.theta - next_to_next_front.theta)
        current_front.depth_cm = current_mass_this_layer / (
            current_front.theta - next_to_next_front.theta
        )
        se = calc_se_from_theta(
            current_front.theta, current_front.theta_e, current_front.theta_r
        )
        current_front.psi_cm = calc_h_from_se(
            se, self.alpha_layer, current_front.m, self.n_layer
        )
        current_front.k_cm_per_h = calc_k_from_se(
            se, current_front.ksat_cm_per_h, current_front.m
        )
        # equivalent to listDeleteFront(next['front_num'])
        self.delete_front(next_front)

    def delete_front(self, front):
        if front.layer_num == self.layer_num:
            for i in range(len(self.wetting_fronts)):
                if self.wetting_fronts[i].is_equal(front):
                    self.wetting_fronts.pop(i)
        else:
            if self.next_layer is not None:
                self.next_layer.delete_front(front)
            else:
                log.error("There is a problem deleting this front")
                raise IndexError

    def wetting_fronts_cross_layer_boundary(self):
        """
        the function lets wetting fronts of a sufficient depth cross layer boundaries; called from lgar_move_wetting_fronts.
        :return:
        """
        layer_fronts = self.get_len_layers()
        m = self.attributes[self.global_params.soil_index["m"]]
        for i in range(layer_fronts):
            extended_neighbors = self.get_extended_neighbors(i)
            current_front = extended_neighbors["current_front"]
            next_front = extended_neighbors["next_front"]
            next_to_next_front = extended_neighbors["next_to_next_front"]
            depth_greater_than_layer = (
                current_front.depth > self.cumulative_layer_thickness
            )
            next_depth_equal_to_thickness = (
                next_front.depth == self.cumulative_layer_thickness
            )
            # TODO VERIFY THAT THIS WORKS!
            # Supposed to not work if the last layer, but that's taken care of before the loop
            if depth_greater_than_layer and next_depth_equal_to_thickness:
                theta_e = self.attributes[
                    self.global_params.soil_index["theta_e"]
                ]
                current_theta = torch.min(theta_e, current_front.theta)
                overshot_depth = current_front.depth - next_front.depth
                # Adding the current wetting front (popped_front) to the next layer
                se = calc_se_from_theta(
                    current_front.theta, current_front.theta_e, current_front.theta_r
                )
                current_front.psi_cm = calc_h_from_se(
                    se, self.alpha_layer, m, self.n_layer
                )
                current_front.ksat_cm_per_h = calc_k_from_se(se, self.ksat_layer, m)
                (
                    extended_neighbors["current_front"],
                    extended_neighbors["next_front"],
                ) = self.recalibrate(
                    current_front,
                    next_front,
                    next_to_next_front,
                    overshot_depth,
                )
        if self.next_layer is not None:
            self.next_layer.wetting_fronts_cross_layer_boundary()
        else:
            return None

    def recalibrate(
        self,
        current_front,
        next_front,
        next_to_next_front,
        overshot_depth,
    ):
        """
        Recalibrating parameters of the wetting front to match the new soil layer
        :param front:
        :param overshot_depth:
        :return:
        """
        # TODO there is a case that if there is too much rainfall we can traverse two layers in one go.
        #  Highly unlikely, but could be worth checking into
        next_theta_r = self.next_layer.attributes[
            self.global_params.soil_index["theta_r"]
        ]
        next_theta_e = self.next_layer.attributes[
            self.global_params.soil_index["theta_e"]
        ]
        next_m = self.next_layer.attributes[
            self.global_params.soil_index["theta_m"]
        ]
        next_alpha = self.next_layer.alpha_layer
        next_n = self.next_layer.n_layer
        theta_new = calc_theta_from_h(
            current_front.psi_cm, next_alpha, next_m, next_n, next_theta_e, next_theta_r
        )
        mbal_correction = overshot_depth * (current_front.theta - next_front.theta)
        mbal_Z_correction = mbal_correction / (
            theta_new - next_to_next_front.theta
        )  # this is the new wetting front depth
        depth_new = (
            self.cumulative_layer_thickness + mbal_Z_correction
        )  # this is the new wetting front absolute depth
        current_front.depth = self.cumulative_layer_thickness
        next_front.theta = theta_new
        next_front.psi_cm = current_front.psi_cm
        next_front.depth = depth_new
        next_front.layer_num = current_front.layer_num + 1
        next_front.dzdt = current_front.dzdt
        current_front.dzdt = torch.tensor(0.0, self.global_params.device)
        current_front.to_bottom = True
        next_front.to_bottom = False
        return current_front, next_front

    def wetting_front_cross_domain_boundary(self):
        """
        the function lets wetting fronts of a sufficient depth interact with the lower boundary; called from lgar_move_wetting_fronts.
        It checks the following case:
        // case : wetting front is the deepest one in the last layer (most deepested wetting front in the domain)
        /**********************************************************/
        :return:
        """
        bottom_flux_cm = torch.tensor(0.0, device=self.global_params.device)
        layer_fronts = self.get_len_layers()
        for i in range(layer_fronts):
            extended_neighbors = self.get_extended_neighbors(i)
            bottom_flux_cm_temp = torch.tensor(0.0, device=self.global_params.device)
            current_front = extended_neighbors["current_front"]
            next_front = extended_neighbors["next_front"]
            next_to_next_front = extended_neighbors["next_to_next_front"]
            if next_to_next_front is None:
                if current_front.depth > self.cumulative_layer_thickness:
                    # this is the water leaving the system through the bottom of the soil
                    bottom_flux_cm_temp = (current_front.theta - next_front.theta) * (
                        current_front.depth - next_front.depth
                    )

                    next_front.theta = current_front.theta
                    se_k = calc_se_from_theta(
                        current_front.theta,
                        current_front.theta_e,
                        current_front.theta_r,
                    )
                    next_front.psi_cm = calc_h_from_se(
                        se_k, self.alpha_layer, current_front.m, self.n_layer
                    )
                    next_front.k_cm_per_h = calc_k_from_se(
                        se_k, self.ksat_layer, current_front.m
                    )
                    _ = self.wetting_fronts.pop(
                        i
                    )  # deleting the current front (i.e front with index i)

            bottom_flux_cm = bottom_flux_cm + bottom_flux_cm_temp
            return bottom_flux_cm

    def fix_dry_over_wet_fronts(self):
        """
        /* The function handles situation of dry over wet wetting fronts
        mainly happen when AET extracts more water from the upper wetting front
        and the front gets drier than the lower wetting front */
        :return:
        """
        mass_change = torch.tensor(0.0, device=self.global_params.device)
        for i in range(len(self.wetting_fronts)):
            neighboring_fronts = self.get_neighboring_fronts(i)
            current_front = neighboring_fronts["current_front"]
            next_front = neighboring_fronts["next_front"]
            if next_front is not None:
                """
                // this part fixes case of upper theta less than lower theta due to AET extraction
                // also handles the case when the current and next wetting fronts have the same theta
                // and are within the same layer
                /***************************************************/
                """
                #  TODO: TEST THIS!
                theta_less = current_front.theta <= next_front.theta
                same_layer = current_front.layer_num == next_front.layer_num
                if theta_less and same_layer:
                    mass_before = self.mass_balance()

                    # replacing current = listDeleteFront(current->front_num);
                    popped_front = self.wetting_fronts.pop(i)

                    # if the dry wetting front is the most surficial then simply track the mass change
                    # due to the deletion of the wetting front;
                    # this needs to be revised
                    if popped_front.layer_num > 0:
                        raise NotImplementedError
                        # se_k = calc_se_from_theta(popped_front.theta, popped_front.theta_e, popped_front.theta_r)
                        # popped_front.psi_cm = calc_h_from_se(se_k, self.alpha_layer, popped_front.m, self.n_layer)
                        # self.update_fronts(popped_front, i)

                    # /* note: mass_before is less when we have wetter front over drier front condition,
                    #  however, lgar_calc_mass_bal returns mass_before > mass_after due to fabs(theta_current - theta_next);
                    # for mass_before the functions compuates more than the actual mass; removing fabs in that function
                    # might be one option, but for now we are adding fabs to mass_change to make sure we added extra water
                    # back to AET after deleting the drier front */
                    mass_after = self.mass_balance()
                    mass_change = mass_change + torch.abs(mass_after - mass_before)
        if self.next_layer is not None:
            return mass_change + self.next_layer.fix_dry_over_wet_fronts()
        else:
            return mass_change

    # def update_fronts(self, popped_front, i):
    #     neighboring_fronts = self.get_neighboring_fronts(i)
    #     for i in range(len(self.wetting_fronts)):
    #
    #         current_local_front = neighboring_fronts["current_front"]
    #         if current_local_front.layer_num < popped_front.layer_num:
    #             se_l = calc_se_from_theta(current_local_front.theta, current_local_front.theta_e, current_local_front.theta_r)
    #             current_local_front.psi_cm = calc_h_from_se(se_l, self.alpha_layer, current_local_front.m, self.n_layer)
    #             current_local_front.theta = calc_theta_from_h(popped_front.psi_cm, self.alpha_layer, current_local_front.m, self.n_layer, current_local_front.theta_e, current_local_front.theta_r)
    #             if neighboring_fronts["next_front"] is not None:
    #                 self.update_fronts(self, popped_front):

    def get_len_layers(self):
        """
        Used to get the number of layers in a wetting front other than the deepest layer
        :return:
        """
        if self.next_layer is not None:
            layer_fronts = len(self.wetting_fronts)
        else:
            # Making sure we don't touch the deepest wetting front
            layer_fronts = len(self.wetting_fronts) - 1
        return layer_fronts

    def update_psi(self):
        """
        Makes sure all pressure values are updated
        :return:
        """
        layer_fronts = self.get_len_layers()
        for i in range(layer_fronts):
            current_front = self.wetting_fronts[i]
            se = calc_se_from_theta(
                current_front.theta, current_front.theta_e, current_front.theta_r
            )
            current_front.psi_cm = calc_h_from_se(
                se, self.alpha_layer, current_front.m, self.n_layer
            )
            current_front.k_cm_per_h = calc_k_from_se(
                se, self.ksat_layer, current_front.m
            )
        if self.next_layer is not None:
            self.next_layer.update_psi()
        else:
            return None

    def calc_dzdt(self, h_p):
        """
        code to calculate velocity of fronts
        equations with full description are provided in the lgar paper (currently under review) */
        :return:
        """
        layer_fronts = self.get_len_layers()
        # We don't calculate dz/dt for the deepest layer
        for i in range(layer_fronts):
            neighboring_fronts = self.get_neighboring_fronts(i)
            current_front = neighboring_fronts["current_front"]
            next_front = neighboring_fronts["next_front"]
            # needed for multi-layered dz/dt equation.  Equal to sum from n=1 to N-1 of (L_n/K_n(theta_n))
            bottom_sum = torch.tensor(0.0, device=self.global_params.device)
            theta_1 = next_front.theta
            theta_2 = current_front.theta
            if current_front.to_bottom:
                current_front.dzdt = torch.tensor(0.0, device=self.global_params.device)
            else:
                if current_front.layer_num > 0:
                    bottom_sum = (
                        bottom_sum
                        + (
                            current_front.depth
                            - self.previous_layer.cumulative_layer_thickness
                        )
                        / current_front.ksat_cm_per_h
                    )
                else:
                    if theta_1 > theta_2:
                        log.error("theta_1 cannot be larger than theta_2")
                        raise ValueError

                geff = calc_geff(
                    self.global_params,
                    self.attributes,
                    theta_1,
                    theta_2,
                    self.alpha_layer,
                    self.n_layer,
                    self.ksat_layer,
                )
                delta_theta = current_front.theta - next_front.theta
                if i == 0 and current_front.layer_num == 0:
                    # This is the top wetting front
                    if delta_theta > 0:
                        dzdt = (
                            1.0
                            / delta_theta
                            * (
                                self.ksat_layer * (geff + h_p) / current_front.depth
                                + current_front.ksat_cm_per_h
                            )
                        )
                    else:
                        dzdt = torch.tensor(0.0, device=self.global_params.device)
                else:  # we are in the second or greater layer
                    raise NotImplementedError
                current_front.dzdt = dzdt
        if self.next_layer is not None:
            self.next_layer.calc_dzdt(h_p)
        else:
            return None

    def move_wetting_fronts(
        self,
        infiltration,
        aet,
        old_mass,
        num_wetting_fronts,
        subtimestep,
        num_wetting_front_count=None,
    ):
        """
        main loop advancing all wetting fronts and doing the mass balance
        loop goes over deepest to top most wetting front
        :param infiltration: The amount of water moving downward in the soil
        :param AET_sub: The amount actual evapotranspiration
        :return:
        """
        if num_wetting_front_count is None:
            # Will only reach this point if the function is called from dpLGAR
            num_wetting_front_count = num_wetting_fronts
        is_bottom_layer = True if self.next_layer is None else False
        # has_many_layers = True if len(self.wetting_fronts) > 1 else False
        # deepest_layer_index = num_wetting_fronts - 1
        volume_infiltraton = torch.tensor(0.0, device=self.global_params.device)
        free_drainage_demand = torch.tensor(0.0, device=self.global_params.device)
        for i in reversed(range(len(self.wetting_fronts))):
            neighboring_fronts = self.get_neighboring_fronts(i)
            if num_wetting_front_count < num_wetting_fronts:
                if (
                    neighboring_fronts["current_front"].is_equal(self.wetting_fronts[-1])
                ):
                    self.deepest_layer_front(neighboring_fronts)
                else:
                    self.wetting_front_in_layer()
            if num_wetting_fronts == self.num_layers and is_bottom_layer:
                self.base_case(infiltration, aet, subtimestep, neighboring_fronts)
            if num_wetting_front_count == 1:
                self.check_column_mass(free_drainage_demand, old_mass, infiltration, aet)
            num_wetting_front_count -= 1
        if self.previous_layer is not None:
            # going to the next layer
            return volume_infiltraton + self.previous_layer.move_wetting_fronts(
                infiltration,
                aet,
                old_mass,
                num_wetting_fronts,
                subtimestep,
                num_wetting_front_count
            )
        else:
            return volume_infiltraton

    def calc_dry_depth(self, subtimestep):
        head_index = 0  # 0 is always the starting index
        current_front = self.wetting_fronts[head_index]

        #  these are the limits of integration
        theta_1 = current_front.theta
        theta_2 = current_front.theta_e
        delta_theta = (
            current_front.theta_e - current_front.theta
        )  # water content of the first (most surficial) existing wetting front
        tau = subtimestep * current_front.ksat_cm_per_h / delta_theta
        geff = calc_geff(
            self.global_params,
            self.attributes,
            theta_1,
            theta_2,
            self.alpha_layer,
            self.n_layer,
            self.ksat_layer,
        )
        # note that dry depth originally has a factor of 0.5 in front
        dry_depth = 0.5 * (tau + torch.sqrt(tau * tau + 4.0 * tau * geff))
        # when dry depth greater than layer 1 thickness, set dry depth to layer 1 thickness
        dry_depth = torch.min(self.cumulative_layer_thickness, dry_depth)
        return dry_depth

    def create_surficial_front(self, dry_depth, ponded_depth, infiltration):
        """
        // ######################################################################################
        /* This subroutine is called iff there is no surfacial front, it creates a new front and
           inserts ponded depth, and will return some amount if can't fit all water */
        // ######################################################################################
        :param dry_depth:
        :param ponded_depth_sub:
        :param infiltration_sub:
        :return:
        """
        to_bottom = False
        head_index = 0
        current_front = self.wetting_fronts[
            head_index
        ]  # Specifically pointing to the first front

        delta_theta = current_front.theta_e - current_front.theta

        if dry_depth * delta_theta > ponded_depth:
            # all the ponded depth enters the soil
            infiltration = ponded_depth
            theta_new = torch.min(
                (current_front.theta + ponded_depth / dry_depth), current_front.theta_e
            )
            new_front = WettingFront(
                self.global_params,
                self.cumulative_layer_thickness,
                self.layer_num,
                self.attributes,
                self.ksat_layer,
            )
            new_front.theta = theta_new
            new_front.depth = dry_depth
            new_front.to_bottom = to_bottom
            # inserting the new front in the front layer
            self.wetting_fronts.insert(0, new_front)
            ponded_depth = torch.tensor(0.0, device=self.global_params.device)
        else:
            # // not all ponded depth fits in
            infiltration = dry_depth * delta_theta
            ponded_depth = ponded_depth - (dry_depth * delta_theta)
            theta_new = current_front.theta
            if dry_depth < self.cumulative_layer_thickness:
                # checking against the first layer
                new_front = WettingFront(
                    self.global_params,
                    self.cumulative_layer_thickness,
                    self.layer_num,
                    self.attributes,
                    self.ksat_layer,
                )
                new_front.depth = dry_depth
                new_front.theta = current_front.theta_e
                new_front.to_bottom = to_bottom
            else:
                new_front = WettingFront(
                    self.global_params,
                    self.cumulative_layer_thickness,
                    self.layer_num,
                    self.attributes,
                    self.ksat_layer,
                )
                new_front.depth = dry_depth
                new_front.theta = current_front.theta_e
                new_front.to_bottom = True
            self.wetting_fronts.insert(0, new_front)

        # These calculations are allowed as we're creating a dry layer of the same soil type
        se = calc_se_from_theta(theta_new, new_front.theta_e, new_front.theta_r)
        new_front.psi_cm = calc_h_from_se(
            se, self.alpha_layer, new_front.m, self.n_layer
        )
        new_front.k_cm_per_h = (
            calc_k_from_se(se, new_front.ksat_cm_per_h, new_front.m)
            * self.global_params.frozen_factor
        )  # // AJ - K_temp in python version for 1st layer
        new_front.dzdt = torch.tensor(0.0, device=self.global_params.device)
        # for now assign 0 to dzdt as it will be computed/updated in lgar_dzdt_calc function

        return ponded_depth, infiltration

    # def set_previous_state(self):
    #     wf = WettingFront(
    #         self.global_params,
    #         self.cumulative_layer_thickness,
    #         self.layer_num,
    #         self.attributes,
    #         self.ksat_layer,
    #     )
    #     # Copying elements of current node to wf
    #     self.wetting_fronts[0].deepcopy(wf)
    #     self.previous_fronts = wf

    def insert_water(
        self,
        subtimestep,
        precip,
        wf_free_drainage_demand,
        ponded_depth,
        infiltration,
    ):
        wf_that_supplies_free_drainage_demand = wf_free_drainage_demand

        f_p = torch.tensor(0.0, device=self.global_params.device)
        runoff = torch.tensor(0.0, device=self.global_params.device)

        # water ponded on the surface
        h_p_ = ponded_depth - precip * subtimestep
        h_p = torch.clamp(h_p_, min=0.0)

        head_index = 0
        (
            current_front,
            current_free_drainage,
            next_free_drainage,
        ) = self.get_drainage_neighbors(0)

        number_of_wetting_fronts = self.calc_num_wetting_fronts()

        layer_num_fp = current_free_drainage.layer_num

        if number_of_wetting_fronts == self.num_layers:
            # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
            geff = torch.tensor(0.0, device=self.global_params.device)
            # i.e., case of no capillary suction, dz/dt is also zero for all wetting fronts
        else:
            # double theta = current_free_drainage->theta;
            theta_1 = next_free_drainage.theta  # theta_below
            theta_2 = current_front.theta_e
            geff = calc_geff(
                self.global_params,
                self.attributes,
                theta_1,
                theta_2,
                self.alpha_layer,
                self.n_layer,
                self.ksat_layer,
            )
        # if the free_drainage wetting front is the top most, then the potential infiltration capacity has the following simple form
        if layer_num_fp == 0:
            f_p = current_front.ksat_cm_per_h * (
                1 + (geff + h_p) / current_free_drainage.depth
            )
        else:
            # This condition has yet to be seen. Leaving this here for last
            raise NotImplementedError
        theta_e1 = current_front.theta
        layer_nums_equal = layer_num_fp == self.num_layers
        thetas_equal = current_free_drainage.theta == theta_e1
        layers_equal_wetting_fronts = self.num_layers == number_of_wetting_fronts
        if layers_equal_wetting_fronts and thetas_equal and layer_nums_equal:
            f_p = torch.tensor(0.0, device=self.global_params.device)
        free_drainage_demand = torch.tensor(0.0, device=self.global_params.device)
        ponded_depth_temp_ = ponded_depth - f_p * subtimestep - free_drainage_demand * 0
        ponded_depth_temp = torch.clamp(ponded_depth_temp_, min=0.0)

        fp_cm = (
            f_p * subtimestep + free_drainage_demand / subtimestep
        )  # infiltration in cm

        if self.global_params.ponded_depth_max_cm > 0.0:
            if ponded_depth_temp < self.global_params.ponded_depth_max_cm:
                runoff = torch.tensor(0.0, device=self.global_params.device)
                infiltration = torch.min(ponded_depth, fp_cm)
                ponded_depth = ponded_depth - infiltration
                # PTL: does this code account for the case where volin_this_timestep can not all infiltrate?
                return runoff, infiltration, ponded_depth
            elif ponded_depth_temp > self.global_params.ponded_depth_max_cm:
                runoff = ponded_depth_temp - self.global_params.ponded_depth_max_cm
                ponded_depth = self.global_params.ponded_depth_max_cm
                infiltration = fp_cm
                return (
                    runoff,
                    infiltration,
                    ponded_depth,
                )
        else:
            # if it got to this point, no ponding is allowed, either infiltrate or runoff
            # order is important here; assign zero to ponded depth once we compute volume in and runoff
            infiltration = torch.min(ponded_depth, fp_cm)
            if ponded_depth < fp_cm:
                runoff = torch.tensor(0.0, device=self.global_params.device)
            else:
                runoff = ponded_depth - infiltration
            ponded_depth_cm = 0.0

        return runoff, infiltration, ponded_depth

    def get_drainage_neighbors(self, i):
        current_front = self.wetting_fronts[i]
        current_free_drainage = self.wf_free_drainage_demand
        if current_free_drainage.layer_num > self.layer_num:
            return self.next_layer.get_drainage_neighbors(i)
        elif current_free_drainage.layer_num < self.layer_num:
            return self.previous_layer.get_drainage_neighbors(i)
        else:
            # The wetting front is in this layer
            for i in range(len(self.wetting_fronts)):
                front = self.wetting_fronts[i]
                if front.is_equal(current_front):
                    if i < len(self.wetting_fronts) - 1:
                        return (
                            current_front,
                            current_free_drainage,
                            self.wetting_fronts[i + 1],
                        )
                    else:
                        return (
                            current_front,
                            current_free_drainage,
                            self.next_layer.wetting_fronts[0],
                        )

    def find_front_layer(self):
        """
        Traverses the soil layers to determine the front layer
        :return: a self obj
        """
        if self.previous_layer is not None:
            return self.previous_layer.find_front_layer()
        else:
            return self

    def print(self, first=True):
        if first:
            log.info(f"[  Depth   Theta          Layer_num   dzdt       ksat      psi   ]")
        for wf in self.wetting_fronts:
            log.info(
                f"[{wf.depth.item():.4f}, {wf.theta.item():.10f},      {wf.layer_num},     {wf.dzdt.item():.6f}, {wf.ksat_cm_per_h.item():.6f}, {wf.psi_cm:.4f}]"
            )
        if self.next_layer is not None:
            return self.next_layer.print(first=False)
        else:
            return None
