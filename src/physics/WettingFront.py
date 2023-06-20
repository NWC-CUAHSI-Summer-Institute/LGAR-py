"""a file to hold code concerning a wetting front"""
import logging

# from src.data.LinkedList import Node

log = logging.getLogger("physics.WettingFront")


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
