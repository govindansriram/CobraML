import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Classification.LogisticModel import LogisticRegression
from GeneralMethods.TrainMethods import train_one_epoch
from GeneralMethods.GeneralDataset import GenericDataSet


class BinaryLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[int],
                 batch_size: int,
                 learning_rate: float,
                 device_name="cpu"):
        self.__device = torch.device(device_name)

        self.__loss_fn = nn.BCELoss().to(device=self.__device)

        data_set = GenericDataSet(torch.tensor(feature_data, dtype=torch.float),
                                  torch.tensor(target_data, dtype=torch.float))

        self.__data_loader = DataLoader(data_set,
                                        batch_size=batch_size,
                                        shuffle=True)

        self.__model = LogisticRegression(len(feature_data[0])).to(self.__device)

        self.__optimizer = optim.Adam(self.__model.parameters(),
                                      lr=learning_rate)

    def fit_model(self, epochs: int) -> torch.tensor:
        mean_epoch_loss = torch.zeros(size=(1, epochs),
                                      dtype=torch.float64,
                                      device=self.__device)

        for epoch in range(epochs):
            train_loss = train_one_epoch(self.__model,
                                         self.__optimizer,
                                         self.__data_loader,
                                         self.__loss_fn,
                                         self.__device)

            mean_epoch_loss[0][epoch] += train_loss

        return mean_epoch_loss

    def evaluate_model(self,
                       feature_tensor: list[list[float]],
                       target_tensor: list[int]) -> torch.tensor:
        self.__model.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(feature_tensor,
                                          dtype=torch.float,
                                          device=self.__device)

            target_tensor = torch.tensor(target_tensor,
                                         dtype=torch.long,
                                         device=self.__device)

            output = self.__model(feature_tensor)

            eq_count = torch.sum(torch.eq(torch.round(output), target_tensor))

            eq_tensor = eq_count / len(target_tensor)

            return eq_tensor

    def get_model(self):
        return self.__model
