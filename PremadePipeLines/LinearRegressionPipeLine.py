import torch
from torch.utils.data import Dataset


class LinearRegressionDataset(Dataset):

    def __init__(self,
                 x_inputs: torch.Tensor,
                 y_targets: torch.Tensor):

        if x_inputs.size()[0] != y_targets.size()[0]:
            raise Exception("row count of input does not match targets")

        self.__inputs = x_inputs
        self.__targets = y_targets

    def __len__(self):
        return len(self.__inputs)

    def __getitem__(self, idx):
        return self.__inputs[idx], self.__targets[idx]
