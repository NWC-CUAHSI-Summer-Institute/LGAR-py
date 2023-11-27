import logging

from omegaconf import DictConfig
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LSTM(nn.Module):
    """
        Taken from https://github.com/jmframe/nash_cascade_neural_network/blob/main/ncn.py
        by @author JMFrame
    """
    def __init__(self, cfg: DictConfig):
        super(LSTM, self).__init__()
        self.cfg = cfg
        # Assuming self.H_tensor and self.theta are already initialized elsewhere
        H_tensor = self.get_the_H_tensor()
        self.input_u_sequence_length = self.n_network_layers + 1

        # Assuming input to the model u is of shape [sequence_length, batch_size, features]
        # LSTM input size is the size of H_tensor plus the number of features in u
        lstm_input_size = H_tensor.numel() + self.input_u_sequence_length  # Here we're assuming u has a single feature dimension
        self.linear = nn.Linear(self.lstm_hidden_size, self.theta.numel())
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers
        )

        # Initialize hidden state and cell state
        self.hidden = (torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size),
                       torch.zeros(self.lstm_num_layers, 1, self.lstm_hidden_size))

    def forward(self):
        raise NotImplementedError

    def _denormalize(self, name: str, param: torch.Tensor) -> torch.Tensor:
        value_range = self.cfg.transformations[name]
        output = (param * (value_range[1] - value_range[0])) + value_range[0]
        return output
