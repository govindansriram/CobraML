from torch import nn
import torch


class LogisticRegression(nn.Module):

    def __init__(self,
                 theta_params: int,
                 device: torch.device):

        super(LogisticRegression, self).__init__()

        self.__device = device

        self.__linear = nn.Linear(theta_params, 1, device=device)

        self.__sigmoid_layer = nn.Sigmoid()

    def forward(self,
                x_input: torch.FloatTensor) -> torch.Tensor:

        x_input = x_input.type(torch.FloatTensor).to(self.__device)

        sigmoid_tensor = self.__sigmoid_layer(self.__linear(x_input))

        return sigmoid_tensor
