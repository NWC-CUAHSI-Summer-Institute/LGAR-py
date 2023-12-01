import logging

import numpy as np
from omegaconf import DictConfig
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dpLGAR.agents.base import BaseAgent
from dpLGAR.data.arid_single_basin import Basin_06332515
from dpLGAR.data.metrics import calculate_nse
from dpLGAR.models.dpLGAR import dpLGAR
from dpLGAR.models.mlp import MLP
from dpLGAR.models.functions.loss import MSE_loss, RangeBoundLoss
from dpLGAR.models.physics.MassBalance import MassBalance
# from dpLGAR.plugins import HybridConfig

log = logging.getLogger(__name__)


class Agent(BaseAgent):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Differentiable LGAR code

        Sets up the initial state of the agent
        :param cfg:
        """
        super().__init__()

        # Setting the cfg object and manual seed for reproducibility
        self.cfg = cfg
        # self.plugin_cfg = hybrid_cfg.nh_config
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float64)

        # Configuring timesteps
        self.cfg.models.endtime_s = (
            self.cfg.models.endtime * self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.subcycle_length_h = self.cfg.models.subcycle_length * (
            1 / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.forcing_resolution_h = (
            self.cfg.models.forcing_resolution / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.time_per_step = (
            self.cfg.models.forcing_resolution_h * self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.nsteps = int(
            self.cfg.models.endtime_s / self.cfg.models.time_per_step
        )
        self.cfg.models.num_subcycles = int(
            self.cfg.models.forcing_resolution_h / self.cfg.models.subcycle_length_h
        )

        self.hourly_mini_batch = int(
            self.cfg.models.hyperparameters.minibatch * 24
        )  # daily to hourly

        self.data = Basin_06332515(self.cfg)
        # self.data_loader = DataLoader(
        #     self.data, batch_size=self.hourly_mini_batch, shuffle=False
        # )
        self.data_loader = DataLoader(
            self.data, batch_size=int(self.cfg.models.endtime), shuffle=False
        )
        self.mlp = MLP(self.cfg)
        # Defining the model and output variables to save
        self.physics_model = dpLGAR(self.cfg)
        self.mass_balance = MassBalance(self.cfg, self.physics_model)

        self.criterion = MSE_loss
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(), lr=self.cfg.models.learning_rate
        )

        self.y_hat = None
        self.y_t = None

        self.current_epoch = 0

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            log.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        self.mlp.initialize_weights()
        self.physics_model.train()
        for epoch in range(1, self.cfg.models.epochs + 1):
            self.train_one_epoch()
            self.current_epoch += 1

            # Resetting the internal states (soil layers) for the next run
            self.physics_model.set_internal_states()
            # Resetting the mass
            self.mass_balance.reset_mass(self.physics_model)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        y_hat_runoff = torch.zeros([int(self.cfg.models.endtime)], device=self.cfg.device)
        y_hat_soil_moisture = torch.zeros([2, int(self.cfg.models.endtime)], device=self.cfg.device)
        y_t_runoff = torch.zeros([int(self.cfg.models.endtime)], device=self.cfg.device)
        y_t_soil_moisture = torch.zeros([2, int(self.cfg.models.endtime)], device=self.cfg.device)
        self.optimizer.zero_grad()
        for i, (
            pet,
            precip,
            _,
            __,
            normalized_attributes,
            streamflow,
            soil_moisture,
        ) in enumerate(self.data_loader):
            alpha, n, ksat, theta_e, theta_r, ponded_depth_max = self.mlp(normalized_attributes[0])
            self.physics_model.initialize(alpha, n, ksat, theta_e, theta_r, ponded_depth_max)
            if i == 0:
                self.mass_balance.starting_volume = self.physics_model.ending_volume
            for j in tqdm(range(int(self.cfg.models.endtime)), desc=f"Epoch {self.current_epoch + 1} Training"):
                with torch.no_grad():
                    _precip = precip[j]
                    _pet = pet[j]
                    runoff, layered_soil_moisture = self.physics_model(_precip, _pet)
                    y_hat_runoff[j] = runoff
                    avg_soil_moisture = [torch.mean(list_) for list_ in layered_soil_moisture]
                    y_hat_soil_moisture[0, j] = avg_soil_moisture[0]
                    y_hat_soil_moisture[1, j] = avg_soil_moisture[1]
                    y_t_runoff[j] = streamflow[j]
                    y_t_soil_moisture[0, j] = soil_moisture[j, 0]
                    y_t_soil_moisture[1, j] = soil_moisture[j, 1]
                    self.mass_balance.change_mass(self.physics_model)  # Updating global balance
                    time.sleep(0.01)
        self.mass_balance.report_mass(self.physics_model)  # Global mass balance
        warmup = self.cfg.models.hyperparameters.warmup
        self.y_hat = torch.stack([y_hat_runoff, y_hat_soil_moisture[0], y_hat_soil_moisture[1]])[:, warmup:]
        self.y_t = torch.stack([y_t_runoff, y_t_soil_moisture[0], y_t_soil_moisture[1]])[:, warmup:]
        self.validate()

    def validate(self) -> None:
        """
        One cycle of model validation
        This function calculates the loss for the given predicted and actual values,
        backpropagates the error, and updates the model parameters.

        Parameters:
        - y_hat_ : The tensor containing predicted values
        - y_t_ : The tensor containing actual values.
        """
        # Outputting trained Nash-Sutcliffe efficiency (NSE) coefficient
        y_hat_streamflow_np = self.y_hat[0].detach().squeeze().numpy()
        y_t_streamflow_np = self.y_t[0].detach().squeeze().numpy()
        log.info(f"trained NSE: {calculate_nse(y_hat_streamflow_np, y_t_streamflow_np):.4}")

        # Compute the overall loss
        loss = self.criterion(self.y_hat, self.y_t)

        # Backpropagate the error
        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()

        # Log the time taken for backpropagation and the calculated loss
        log.info(f"Back prop took : {(end - start):.6f} seconds")
        log.info(f"Loss: {loss}")

        # Update the model parameters
        self.optimizer.step()

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError
