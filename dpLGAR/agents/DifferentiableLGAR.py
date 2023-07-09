import logging
from omegaconf import DictConfig
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dpLGAR.agents.base import BaseAgent
from dpLGAR.data.Data import Data
from dpLGAR.data.metrics import calculate_nse
from dpLGAR.models.dpLGAR import dpLGAR
from dpLGAR.models.physics.MassBalance import MassBalance

log = logging.getLogger("agents.DifferentiableLGAR")


class DifferentiableLGAR(BaseAgent):
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the Differentiable LGAR code

        Sets up the initial state of the agent
        :param cfg:
        """
        super().__init__()

        # Setting the cfg object and manual seed for reproducibility
        self.cfg = cfg
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

        self.hourly_mini_batch = (
            cfg.models.hyperparameters.minibatch * 24
        )  # daily to hourly
        # Defining the torch Dataset and Dataloader
        self.data = Data(self.cfg)
        self.data_loader = DataLoader(
            self.data, batch_size=self.hourly_mini_batch, shuffle=False
        )

        # Defining the model and output variables to save
        self.model = dpLGAR(self.cfg)
        self.percolation_output = torch.zeros(
            [self.cfg.models.nsteps], device=self.cfg.device
        )
        self.mass_balance = MassBalance(cfg, self.model)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.models.hyperparameters.learning_rate
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
        self.model.train()
        for epoch in range(1, self.cfg.models.hyperparameters.epochs + 1):
            self.train_one_epoch()
            self.current_epoch += 1

            # Resetting the internal states (soil layers) for the next run
            self.model.set_internal_states()
            # Resetting the mass
            self.mass_balance.reset_mass(self.model)

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        for i, (x, y_t) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            # Resetting output vars
            self.y_hat = torch.zeros([x.shape[0]], device=self.cfg.device)  # runoff
            self.y_t = y_t
            self.percolation_output = torch.zeros([x.shape[0]], device=self.cfg.device)
            for j in trange(x.shape[0], desc=f"Running Minibatch {i+1}", leave=True):
                # Minibatch loop
                inputs = x[j]
                runoff, percolation = self.model(inputs)
                self.y_hat[j] = runoff
                self.percolation_output[j] = percolation
                # Updating the total mass of the system
                self.mass_balance.change_mass(self.model)
                time.sleep(0.01)
            # self.mass_balance.report_mass(self.model)
            if self.y_hat.requires_grad:
                if i == 0:
                    warmup = self.cfg.models.hyperparameters.warmup
                    self.y_hat = self.y_hat[warmup:]
                    self.y_t = self.y_t[warmup:]
                # If there is no gradient (i.e. no runoff), then we shouldn't validate
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
        y_hat_np = self.y_hat.detach().squeeze().numpy()
        y_t_np = self.y_t.detach().squeeze().numpy()
        log.info(
            f"trained NSE: {calculate_nse(y_hat_np, y_t_np):.4}"
        )

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
        # self.model.update_soil_parameters()

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
