import logging
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn

from lgartorch.models.physics.layers.WettingFront import WettingFront
from lgartorch.models.physics.lgar.aet import calc_aet
from lgartorch.models.physics.utils import (
    calc_theta_from_h,
    calc_se_from_theta,
    calc_h_from_se,
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
        self.wf_free_drainage_demand = (
            self.layer_num
        )  # TODO FIX THIS. THIS IS NOT IMPLEMENTED
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
        # TODO CHECK IF THE OBJECT POINTERS NEED TO BE COPIED VS THE OBJ
        state = {}
        state["wetting_fronts"] = []
        for i in range(len(self.wetting_fronts)):
            state["wetting_fronts"].append(
                WettingFront(
                    self.global_params,
                    self.cumulative_layer_thickness,
                    self.layer_num,
                    self.attributes,
                    self.ksat_layer,
                )
            )
            state["wetting_fronts"][i].deepcopy(WettingFront)
        # try:
        #     state["previous_layer"] = self.previous_layer.deepcopy()
        # except AttributeError:
        #     pass
        # try:
        #     state["next_layer"] = self.next_layer.deepcopy()
        # except AttributeError:
        #     pass
        return state

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
        m = self.attributes[self.global_params.soil_property_indexes["m"]]
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_property_indexes["theta_r"]]
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
        if self.previous_layer is None:
            return new_mass, prior_mass, delta_thetas, delta_thickness
        else:
            return self.previous_layer.populate_delta_thickness(
                psi_cm_old, psi_cm, prior_mass, new_mass, delta_thetas, delta_thickness
            )

    def recalculate_mass(self, psi_cm, new_mass, delta_thetas, delta_thickness):
        theta_e = self.attributes[self.global_params["theta_e"]]
        theta_r = self.attributes[self.global_params["theta_r"]]
        m = self.attributes[self.global_params["m"]]
        theta_layer = calc_theta_from_h(
            psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
        )
        new_mass = new_mass + delta_thickness[self.layer_num] * (
            theta_layer - delta_thetas[self.layer_num]
        )
        if self.next_layer is None:
            return new_mass
        else:
            return self.next_layer.recalculate_mass(
                psi_cm, new_mass, delta_thetas, delta_thickness
            )

    def theta_mass_balance(
        self,
        psi_cm,
        new_mass,
        prior_mass,
        delta_thetas,
        delta_thickness,
    ):
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_property_indexes["theta_r"]]
        m = self.attributes[self.global_params.soil_property_indexes["m"]]
        delta_mass = torch.abs(new_mass - prior_mass)
        tolerance = 1e-12
        factor = 1.0
        # flag that determines capillary head to be incremented or decremented
        switched = False
        theta = torch.tensor(
            0.0, device=self.global_params.device
        )  # this will be updated and returned
        psi_cm_loc_prev = psi_cm
        delta_mass_prev = delta_mass
        count_no_mass_change = torch.tensor(0.0, device=self.global_params.device)
        break_no_mass_change = torch.tensor(5.0, device=self.global_params.device)

        # check if the difference is less than the tolerance
        if delta_mass <= tolerance:
            theta = calc_theta_from_h(
                psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
            )
            return theta

        # the loop increments/decrements the capillary head until mass difference between
        # the new and prior is within the tolerance
        while delta_mass > tolerance:
            if new_mass > prior_mass:
                psi_cm = psi_cm + 0.1 * factor
                switched = False
            else:
                if not switched:
                    switched = True
                    factor = factor * 0.1

                psi_cm_loc_prev = psi_cm
                psi_cm_loc = psi_cm - 0.1 * factor

                if psi_cm < 0 and psi_cm_loc_prev != 0:
                    psi_cm = psi_cm_loc_prev * 0.1

            theta = calc_theta_from_h(
                psi_cm, self.alpha_layer, self.n_layer, m, theta_e, theta_r
            )
            new_mass = delta_thickness[self.layer_num] * (
                theta - delta_thetas[self.layer_num]
            )

            new_mass = self.recalculate_mass(
                psi_cm, new_mass, delta_thetas, delta_thickness
            )
            delta_mass = torch.abs(new_mass - prior_mass)

            # stop the loop if the error between the current and previous psi is less than 10^-15
            # 1. enough accuracy, 2. the algorithm can't improve the error further,
            # 3. avoid infinite loop, 4. handles a corner case when prior mass is tiny (e.g., <1.E-5)
            # printf("A1 = %.20f, %.18f %.18f %.18f %.18f \n ",fabs(psi_cm_loc - psi_cm_loc_prev) , psi_cm_loc, psi_cm_loc_prev, factor, delta_mass);
            if torch.abs(psi_cm_loc - psi_cm_loc_prev) < 1e-15 and factor < 1e-13:
                break
            if torch.abs(delta_mass - delta_mass_prev) < 1e-15:
                count_no_mass_change += 1
            else:
                count_no_mass_change = 0
            if count_no_mass_change == break_no_mass_change:
                break
            if psi_cm_loc <= 0 and psi_cm_loc_prev < 1e-50:
                break
            delta_mass_prev = delta_mass
        return theta

    def base_case(self, i, percolation, aet, subtimestep, neighboring_fronts):
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

        new_mass, prior_mass, delta_thetas, delta_thickness = self.previous_layer.populate_delta_thickness(
            psi_cm_old, psi_cm, prior_mass, new_mass, delta_thetas, delta_thickness
        )
        delta_thickness[self.layer_num] = current_front.depth - base_thickness

        free_drainage_demand = 0
        # TODO FIX THIS!!!!! NEED AN ANSWER ABOUT WF_FREE_DRAINAGE
        if self.wf_free_drainage_demand == self.layer_num:
            prior_mass = prior_mass + percolation - (free_drainage_demand + aet)

        # theta mass balance computes new theta that conserves the mass; new theta is assigned to the current wetting front
        theta_new = self.theta_mass_balance(
            psi_cm,
            new_mass,
            prior_mass,
            delta_thetas,
            delta_thickness,
        )
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_property_indexes["theta_r"]]
        m = self.attributes[self.global_params.soil_property_indexes["m"]]
        current_front.theta = torch.minimum(theta_new, theta_e)
        se = calc_se_from_theta(current_front.theta, theta_e, theta_r)
        current_front.psi_cm = calc_h_from_se(se, self.alpha_layer, m, self.n_layer)

    def deepest_layer_front(self):
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
        :return:
        """
        pass

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

    def top_layer_in_front(self):
        """
                // if f_p (predicted infiltration) causes theta > theta_e, mass correction is needed.
        // depth of the wetting front is increased to close the mass balance when theta > theta_e.
        // l == 1 is the last iteration (top most wetting front), so do a check on the mass balance)
        // this part should be moved out of here to a subroutine; add a call to that subroutine
            :return:
        """
        raise NotImplementedError

    def get_neighboring_fronts(self, i):
        """
        Gets the current, previous, and next fronts for the current model state
        Also gets the previous state's current and next fronts
        :param i:
        :return:
        """
        get_neighboring_fronts = {}
        get_neighboring_fronts["current_front"] = self.wetting_fronts[i]
        get_neighboring_fronts["previous_current_front"] = self.previous_state[
            "wetting_fronts"
        ][i]
        get_neighboring_fronts["next_front"] = None
        get_neighboring_fronts["previous_front"] = None
        get_neighboring_fronts["previous_next_front"] = None
        if i > 0:
            get_neighboring_fronts["previous_front"] = self.wetting_fronts[i - 1]
        if i < (len(self.wetting_fronts) - 1):
            get_neighboring_fronts["next_front"] = self.wetting_fronts[i + 1]
            get_neighboring_fronts["previous_next_front"] = self.previous_state[
                "wetting_fronts"
            ][i + 1]
        if self.next_layer is not None:
            if i == (len(self.wetting_fronts) - 1):
                get_neighboring_fronts["next_front"] = self.next_layer.wetting_fronts[0]
                get_neighboring_fronts["previous_next_front"] = self.previous_state[
                    "wetting_fronts"
                ][0]
        if self.previous_layer is not None:
            if i == 0:
                get_neighboring_fronts[
                    "previous_front"
                ] = self.previous_layer.wetting_fronts[-1]
        return get_neighboring_fronts

    def move_wetting_fronts(self, percolation, aet, num_wetting_fronts, subtimestep):
        """
        main loop advancing all wetting fronts and doing the mass balance
        loop goes over deepest to top most wetting front
        :param percolation: The amount of water moving downward in the soil
        :param AET_sub: The amount actual evapotranspiration
        :return:
        """
        is_bottom_layer = True if self.next_layer is None else False
        deepest_layer_index = len(self.wetting_fronts) - 1
        column_depth = self.global_params.cum_layer_thickness[-1]
        volume_infiltraton = torch.tensor(0.0, device=self.global_params.device)
        for i in reversed(range(len(self.wetting_fronts))):
            neighboring_fronts = self.get_neighboring_fronts(i)
            if i == deepest_layer_index and self.next_layer is not None:
                # Finding the deepest layer that is not the bottom layer
                self.deepest_layer_front()
            if num_wetting_fronts == self.num_layers:
                self.base_case(i, percolation, aet, subtimestep, neighboring_fronts)
            if i < len(self.wetting_fronts):
                self.wetting_front_in_layer()
            if i == 0 and self.layer_num == 0:
                self.top_layer_front()
        return volume_infiltraton

    def calc_aet(self, pet: Tensor) -> Tensor:
        """
        ONLY CALLED FROM TOP LAYER
        Calculates the Actual Evapotranspiration for each layer
        :param pet: Potential evapotranspiration
        :param subcycle_length_h: the length of each subcycle step (in hours)
        :return:
        """
        top_wetting_front = self.wetting_fronts[0]
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
        theta_r = self.attributes[self.global_params.soil_property_indexes["theta_r"]]
        m = self.attributes[self.global_params.soil_property_indexes["m"]]
        aet = calc_aet(
            self.global_params,
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
        theta_e = self.attributes[self.global_params.soil_property_indexes["theta_e"]]
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
            # TODO TEST THIS!!!
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

    def get_deeper_fronts(self, i):
        if i < len(self.wetting_fronts) - 1:
            next_front = self.wetting_fronts[i + 1]
            next_to_next_front = self.next_layer.wetting_fronts[0]
        else:
            next_front = self.next_layer.wetting_fronts[0]
            if len(self.next_layer.wetting_fronts) == 1:
                try:
                    next_to_next_front = self.next_layer.next_layer.wetting_fronts[0]
                except AttributeError:
                    # No more layers left
                    next_to_next_front = None
            else:
                next_to_next_front = self.next_layer.wetting_fronts[1]
        return next_front, next_to_next_front

    def is_passing(self, current_front, next_front):
        """
        # case : wetting front passing another wetting front within a layer
        :param current_front: the current wetting front
        :param next_front: the next wetting front
        :return:
        """
        larger_depth = current_front.depth > next_front.depth
        same_layer = current_front.layer_num == next_front.layer_num
        return larger_depth and same_layer and not next_front.to_bottom

    def merge_wetting_fronts(self):
        for i in range(len(self.wetting_fronts)):
            current_front = self.wetting_fronts[i]
            try:
                next_front = self.wetting_fronts[i + 1]
                next_to_next_front = self.wetting_fronts[i + 2]
            except IndexError:
                next_front, next_to_next_front = self.get_deeper_fronts(i)
            if self.is_passing():
                raise NotImplementedError

    def wetting_fronts_cross_layer_boundary(self):
        raise NotImplementedError

    def wetting_front_cross_domain_boundary(self):
        raise NotImplementedError

    def fix_dry_over_wet_fronts(self):
        raise NotImplementedError

    def update_psi(self):
        raise NotImplementedError

    def calc_dzdt(self):
        raise NotImplementedError

    def giuh_runoff(self):
        raise NotImplementedError
