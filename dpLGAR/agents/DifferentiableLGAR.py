import logging
from omegaconf import DictConfig
import time
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm, trange

from dpLGAR.agents.base import BaseAgent
from dpLGAR.data.Data import Data
from dpLGAR.data.graphdatasampler import GraphDataSampler
from dpLGAR.data.metrics import calculate_nse
from dpLGAR.models.dpLGAR import dpLGAR
from dpLGAR.models.functions.loss import MSE_loss, RangeBoundLoss
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

        # Initialize DistributedDataParallel (DDP)
        self.rank = int(cfg.local_rank)
        log.info(f"Initializing Distributed Data: {self.rank}")
        self.setup(self.cfg)

        # Configuring subcycles (If we're running hourly, 15 mins, etc)
        self.cfg.models.subcycle_length_h = self.cfg.models.subcycle_length * (
            1 / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.forcing_resolution_h = (
            self.cfg.models.forcing_resolution / self.cfg.conversions.hr_to_sec
        )
        self.cfg.models.num_subcycles = int(
            self.cfg.models.forcing_resolution_h / self.cfg.models.subcycle_length_h
        )

        # Setting the number of values per batch (Currently we want this to be 1)
        self.hourly_mini_batch = int(cfg.models.hyperparameters.minibatch * 24)

        # Defining the torch Dataset and Dataloader
        self.data = Data(self.cfg)
        self.data_size = int(self.data.x.shape[0] / cfg.nproc)
        self.graph_sampler = GraphDataSampler(
            dataset=self.data, batch_size=self.data_size, shuffle=False
        )
        self.data_loader = DataLoader(
            self.data,
            batch_size=self.hourly_mini_batch,
            sampler=self.graph_sampler,
            shuffle=False,
        )

        # Defining the model and output variables to save
        self.model = dpLGAR(self.cfg, self.data.soil_information)
        self.mass_balance = MassBalance(cfg, self.model)

        self.criterion = MSE_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.models.hyperparameters.learning_rate
        )

        lb = cfg.models.hyperparameters.lb
        ub = cfg.models.hyperparameters.ub
        self.range_bound_loss = RangeBoundLoss(lb, ub, factor=1.0)

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
            interrupt = True
            self.finalize(interrupt)
            log.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        self.model.train()
        self.net = DDP(self.model)
        for epoch in range(1, self.cfg.models.hyperparameters.epochs + 1):
            if self.rank == 0:
                log.info(f"Running epoch: {self.current_epoch}")
                log.info(f"-----Current Params-----")
            self.model.print_params()
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
        self.graph_sampler.set_epoch(self.current_epoch)
        y_hat_ = torch.zeros([len(self.data_loader)], device=self.cfg.device)  # runoff
        y_t_ = torch.zeros([len(self.data_loader)], device=self.cfg.device)  # runoff
        self.optimizer.zero_grad()
        for i, (x, y_t) in enumerate(
            tqdm(
                self.data_loader,
                desc=f"Rank: {self.rank} Epoch {self.current_epoch + 1} Training",
            )
        ):
            # Resetting output vars
            runoff = self.net(i, x.squeeze())
            y_hat_[i] = runoff
            y_t_[i] = y_t
            # Updating the total mass of the system
            self.mass_balance.change_mass(self.model)  # Updating global balance
            time.sleep(0.01)
        self.mass_balance.report_mass(self.model)  # Global mass balance
        warmup = self.cfg.models.hyperparameters.warmup
        self.y_hat = y_hat_[warmup:]
        self.y_t = y_t_[warmup:]
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
        log.info(f"trained NSE: {calculate_nse(y_hat_np, y_t_np):.4}")

        # Compute the overall loss
        loss_mse = self.criterion(self.y_hat, self.y_t)

        # Compute the range bound loss for the parameters you want to constrain
        params = [
            self.model.alpha,
            self.model.n,
            self.model.ksat,
            self.model.ponded_depth_max,
        ]
        bound_loss = self.range_bound_loss(params)

        loss = loss_mse + bound_loss

        # Backpropagate the error
        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()

        # Log the time taken for backpropagation and the calculated loss
        log.info(f"Back prop took : {(end - start):.6f} seconds")
        log.info(f"Loss: {loss}")

        # Update the model parameters
        self.optimizer.step()
        # torch.autograd.set_detect_anomaly(True)

    def finalize(self, interrupt=False):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        if self.current_epoch > 0:
            from pathlib import Path

            dir_path = Path(self.cfg.models.save_path)
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
            if interrupt:
                file_path = dir_path / self.cfg.models.not_finished_name.format(
                    self.current_epoch
                )
            else:
                file_path = dir_path / self.cfg.models.save_name
            torch.save(self.model.state_dict(), file_path)
        self.cleanup()

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

    def setup(self, cfg: DictConfig) -> None:

        dist_url = "env://"  # Not important for what we're doing
        dist.init_process_group(backend="gloo")
        dist.barrier()

    def cleanup(self):
        dist.destroy_process_group()
