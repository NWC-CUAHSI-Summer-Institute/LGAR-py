"""a file to hold code concerning a wetting front"""
import logging

# from src.data.LinkedList import Node

log = logging.getLogger("physics.WettingFront")


class WettingFront():
    def __init__(self, depth, theta, layer_num, bottom_flag):
        super().__init__() # Use front_num as val for the base Node class
        self.depth_cm = depth
        self.theta = theta
        self.layer_num = layer_num
        self.to_bottom = bottom_flag
        self.dzdt_cm_per_h = 0.0
        self.psi_cm = None
        self.k_cm_per_h = None
