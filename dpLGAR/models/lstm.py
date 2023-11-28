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
        num_attributes = len(self.cfg.data.varC)
        num_forcings = len(self.cfg.data.varT)

        # Assuming input to the model u is of shape [sequence_length, batch_size, features]
        # LSTM input size is the size of H_tensor plus the number of features in u
        # lstm_input_size = num_attributes + num_forcings
        lstm_input_size = num_forcings
        lstm_hidden_size = self.cfg.model.hidden_size
        output_size = len(self.cfg.model.target_variables)
        lstm_num_layers = self.cfg.model.num_lstm_layers
        self.linear = nn.Linear(lstm_hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers
        )
        self.hidden = (torch.zeros(lstm_num_layers, 1, lstm_hidden_size),
                       torch.zeros(lstm_num_layers, 1, lstm_hidden_size))

    def forward(self, x):
        #TODO
        # Prepare LSTM inputs
        # run LSTM
        # denormalize parameters
        # make sure we can update with dpLGAR's soil parameters

        # Ensure lstm_input is [1, sequence_length, input_size]
        lstm_input = x.unsqueeze(0)

        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input, self.hidden)
        lstm_output = self.linear(lstm_output)  # Get the last time step output for the linear layer
        lstm_output = self.sigmoid(lstm_output)  # Apply the sigmoid activation
        # Post-process LSTM output to match the shape of self.theta
        # Assuming lstm_output is [1, sequence_length, hidden_size] and self.theta is [hidden_size]
        lstm_output = lstm_output.squeeze(0)  # remove batch dimension
        self.theta = lstm_output[-1]  # take the last timestep

    def _denormalize(self, name: str, param: torch.Tensor) -> torch.Tensor:
        value_range = self.cfg.transformations[name]
        output = (param * (value_range[1] - value_range[0])) + value_range[0]
        return output
