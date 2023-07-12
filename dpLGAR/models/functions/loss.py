"""Define loss functions here"""

import torch
from torch import nn


MSE_loss = nn.MSELoss()


class RangeBoundLoss(nn.Module):
    """limit parameters from going out of range"""
    def __init__(self, lb, ub, factor=1.0):
        super(RangeBoundLoss, self).__init__()
        self.lb = torch.tensor(lb)
        self.ub = torch.tensor(ub)
        self.factor = torch.tensor(factor)

    def forward(self, params):
        loss = torch.tensor(0.0, dtype=torch.float64)
        # Process the ParameterList objects
        for i in range(len(params) - 1):  # subtract 1 to exclude the last Parameter
            lb = self.lb[i]
            ub = self.ub[i]
            params_tensor = torch.stack([param for param in params[i]])  # convert ParameterList to tensor
            upper_bound_loss = torch.sum(self.factor * torch.relu(params_tensor - ub))
            lower_bound_loss = torch.mean(self.factor * torch.relu(lb - params_tensor))
            loss = loss + upper_bound_loss + lower_bound_loss

        # # Process the last Parameter separately
        # lb = self.lb[-1]
        # ub = self.ub[-1]
        # params_tensor = params[-1]  # this is already a tensor
        # upper_bound_loss = self.factor * torch.relu(params_tensor - ub)
        # lower_bound_loss = self.factor * torch.relu(lb - params_tensor)
        # loss = loss + upper_bound_loss + lower_bound_loss
        return loss