import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Classification.LogisticModel import LogisticRegression
from GeneralMethods.TrainMethods import train_one_epoch, train_one_epoch_lbfgs
from GeneralMethods.GeneralDataset import GenericDataSet


class BinaryLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[int],
                 batch_size: int,
                 learning_rate: float,
                 device_name="cpu",
                 optimizer_name="ADAM",
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 amsgrad=False,
                 momentum=0,
                 weight_decay=0,
                 dampening=0,
                 nesterov=False,
                 max_iter=20,
                 max_eval=25,
                 tolerance_grad=1e-5,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None):

        self.__device = torch.device(device_name)

        self.__loss_fn = nn.BCELoss().to(device=self.__device)

        data_set = GenericDataSet(torch.tensor(feature_data, dtype=torch.float),
                                  torch.tensor(target_data, dtype=torch.float))

        self.__data_loader = DataLoader(data_set,
                                        batch_size=batch_size,
                                        shuffle=True)

        self.__model = LogisticRegression(len(feature_data[0])).to(self.__device)

        if optimizer_name == "ADAM":
            self.__optimizer = optim.Adam(self.__model.parameters(),
                                          lr=learning_rate,
                                          betas=betas,
                                          eps=eps,
                                          weight_decay=weight_decay,
                                          amsgrad=amsgrad)
        elif optimizer_name == "SGD":
            self.__optimizer = optim.SGD(self.__model.parameters(),
                                         lr=learning_rate,
                                         momentum=momentum,
                                         weight_decay=weight_decay,
                                         dampening=dampening,
                                         nesterov=nesterov)
        elif optimizer_name == "LBFGS":
            self.__optimizer = torch.optim.LBFGS(self.__model.parameters(),
                                                 lr=learning_rate,
                                                 max_iter=max_iter,
                                                 max_eval=max_eval,
                                                 tolerance_grad=tolerance_grad,
                                                 tolerance_change=tolerance_change,
                                                 history_size=history_size,
                                                 line_search_fn=line_search_fn)

    def fit_model(self, epochs: int) -> torch.tensor:
        mean_epoch_loss = torch.zeros(size=(1, epochs),
                                      dtype=torch.float64,
                                      device=self.__device)

        if isinstance(self.__optimizer, torch.optim.Adam) or isinstance(self.__optimizer, torch.optim.SGD):
            for epoch in range(epochs):
                train_loss = train_one_epoch(self.__model,
                                             self.__optimizer,
                                             self.__data_loader,
                                             self.__loss_fn,
                                             self.__device)

                mean_epoch_loss[0][epoch] += train_loss

        else:
            for epoch in range(epochs):
                train_loss = train_one_epoch_lbfgs(self.__model,
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

            eq_tensor = eq_count / target_tensor.size()[0]

            return eq_tensor

    def get_model(self):
        return self.__model
