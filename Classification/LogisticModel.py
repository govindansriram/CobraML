from torch import nn
import torch


class LogisticRegression(nn.Module):

    def __init__(self,
                 theta_params: int):

        super(LogisticRegression, self).__init__()
        self.__linear = nn.Linear(theta_params, 1)
        self.__sigmoid_layer = nn.Sigmoid()

    def forward(self,
                x_input: torch.tensor) -> torch.tensor:

        return self.__sigmoid_layer(self.__linear(x_input))
