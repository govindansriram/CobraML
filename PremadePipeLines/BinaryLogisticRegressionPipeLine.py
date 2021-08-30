import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from Classification.LogisticModel import LogisticRegression
from GeneralMethods.TrainMethods import train_one_epoch


class LogisticRegressionDataset(Dataset):

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


class BinaryLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[list[int]],
                 batch_size: int,
                 learning_rate: float,
                 device_name="cpu"):
        self.__device = torch.device(device_name)

        self.__loss_fn = nn.BCELoss().to(device=self.__device)

        data_set = LogisticRegressionDataset(torch.tensor(feature_data,
                                                          device=self.__device,
                                                          dtype=torch.float64),
                                             torch.tensor(target_data,
                                                          device=self.__device,
                                                          dtype=torch.float64))

        self.__train_data_loader = DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=True)

        self.__model = LogisticRegression(len(feature_data[0]), self.__device).to(self.__device)

        self.__optimizer = optim.Adam(self.__model.parameters(),
                                      lr=learning_rate)

    def fit_model(self, epochs: int) -> torch.Tensor:
        mean_epoch_loss = torch.zeros(size=(1, epochs),
                                      dtype=torch.float64,
                                      device=self.__device)

        for epoch in range(epochs):
            train_loss = train_one_epoch(self.__model,
                                         self.__optimizer,
                                         self.__train_data_loader,
                                         self.__loss_fn,
                                         self.__device)

            mean_epoch_loss[0][epoch] += train_loss

        return mean_epoch_loss

    def evaluate_model(self,
                       feature_tensor: list[list[float]],
                       target_tensor: list[list[int]]) -> torch.Tensor:
        self.__model.eval()
        with torch.no_grad():

            feature_tensor = torch.tensor(feature_tensor, dtype=torch.float64, device=self.__device)
            target_tensor = torch.tensor(target_tensor, dtype=torch.float64, device=self.__device)

            output = self.__model(feature_tensor)

            eq_count = torch.sum(torch.eq(torch.round(output), target_tensor))

            eq_tensor = eq_count / len(target_tensor)

            return eq_tensor

    def get_model(self):
        return self.__model
