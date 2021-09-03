from torch import nn
import torch


class MultiClassLogisticRegression(nn.Module):

    def __init__(self,
                 theta_params: int,
                 num_of_classes: int):
        super(MultiClassLogisticRegression, self).__init__()

        self.__linear = nn.Linear(theta_params, num_of_classes)

    def forward(self,
                x_input: torch.tensor) -> torch.tensor:

        return self.__linear(x_input)
