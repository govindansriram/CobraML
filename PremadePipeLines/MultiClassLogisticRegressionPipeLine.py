import torch
import torch.nn.functional as F
from Classification.MultiClassLogisticModel import MultiClassLogisticRegression as MCLR
from GeneralMethods.GeneralDataset import GenericDataSet
from GeneralMethods.GeneralOptimizer import get_optimizer
from GeneralMethods.TrainMethods import train_one_epoch, train_one_epoch_lbfgs
from torch.utils.data import DataLoader


class MultiClassLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[list[int]],
                 batch_size: int,
                 learning_rate: float,
                 num_output_classes: int,
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

        self.__loss_fn = torch.nn.CrossEntropyLoss().to(self.__device)

        self.__model = MCLR(len(feature_data[0]), num_output_classes).to(self.__device)

        data_set = GenericDataSet(torch.tensor(feature_data, dtype=torch.float),
                                  torch.tensor(target_data, dtype=torch.long))

        self.__data_loader = DataLoader(data_set, batch_size, shuffle=True)

        self.__optimizer = get_optimizer(self.__model,
                                         learning_rate,
                                         optimizer_name=optimizer_name,
                                         betas=betas,
                                         eps=eps,
                                         amsgrad=amsgrad,
                                         momentum=momentum,
                                         weight_decay=weight_decay,
                                         dampening=dampening,
                                         nesterov=nesterov,
                                         max_iter=max_iter,
                                         max_eval=max_eval,
                                         tolerance_grad=tolerance_grad,
                                         tolerance_change=tolerance_change,
                                         history_size=history_size,
                                         line_search_fn=line_search_fn)

    def fit_model(self, epochs: int):
        mean_epoch_loss = torch.zeros(size=(1, epochs),
                                      dtype=torch.float,
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

    def validate_model(self,
                       feature_data: list[list[float]],
                       target_data: list[list[int]]) -> torch.Tensor:
        with torch.no_grad():
            feature_tensor = torch.tensor(feature_data,
                                          dtype=torch.float,
                                          device=self.__device)

            target_tensor = torch.tensor(target_data,
                                         dtype=torch.int,
                                         device=self.__device)

            output = F.log_softmax(self.__model(feature_tensor), dim=1)

            index_tensor = torch.unsqueeze(torch.max(output, dim=1)[1], dim=1)

            eq_count = torch.sum(torch.eq(index_tensor, torch.unsqueeze(target_tensor, dim=1)))

            return eq_count / target_tensor.size()[0]
