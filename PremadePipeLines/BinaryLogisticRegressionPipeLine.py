import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from Classification.LogisticModel import LogisticRegression
from GeneralMethods.TrainMethods import train_one_epoch_lbfgs, train_one_epoch


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
                 momentum: float,
                 train_split=0.7,
                 randomize=False,
                 device_name="cpu",
                 optimizer="SGD"):

        optim_list = ("SGD",
                      "ADAM",
                      "LBFGS")

        if optimizer not in optim_list:
            raise Exception("Invalid optimizer")

        if train_split >= 1:
            raise Exception("split exceeds 100%")

        data = list(zip(feature_data, target_data))
        random.shuffle(data)

        feature_list, target_list = list(zip(*data))
        feature_list = list(feature_list)
        target_list = list(target_list)

        self.__loss_fn = nn.BCELoss()

        self.__device = torch.device(device_name)

        cut_off_num = round(len(feature_list) * train_split)
        cut_off_num_two = cut_off_num + (len(feature_list) - cut_off_num) // 2

        train_feature_list = feature_list[0: cut_off_num]
        train_target_list = target_list[0: cut_off_num]

        test_feature_list = feature_list[cut_off_num: cut_off_num_two]
        test_target_list = target_list[cut_off_num: cut_off_num_two]

        self.__val_feature_tensor = feature_list[cut_off_num_two: len(feature_list)]
        self.__val_target_tensor = target_list[cut_off_num_two: len(feature_list)]

        data_set = LogisticRegressionDataset(torch.tensor(train_feature_list,
                                                          device=self.__device).type(torch.DoubleTensor),
                                             torch.tensor(train_target_list,
                                                          device=self.__device).type(torch.DoubleTensor))

        self.__train_data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        data_set = LogisticRegressionDataset(torch.tensor(test_feature_list,
                                                          device=self.__device).type(torch.DoubleTensor),
                                             torch.tensor(test_target_list,
                                                          device=self.__device).type(torch.DoubleTensor))

        self.__test_data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        self.__model = LogisticRegression(len(train_feature_list[0]),
                                          device=device_name,
                                          randomize=randomize).to(self.__device)

        if optimizer == "SGD":
            self.__optimizer = optim.SGD([self.__model.get_parameters()],
                                         lr=learning_rate,
                                         momentum=momentum)
        elif optimizer == "ADAM":
            self.__optimizer = optim.Adam([self.__model.get_parameters()],
                                          lr=learning_rate)

        else:
            self.__optimizer = optim.LBFGS([self.__model.get_parameters()],
                                           lr=learning_rate)

    def fit_model(self, epochs: int) -> tuple[list, [dict[str: float]]]:

        train_loss_list = []
        test_loss_list = []

        for epoch in range(epochs):
            if isinstance(self.__optimizer, optim.LBFGS):
                train_loss = train_one_epoch_lbfgs(self.__model,
                                                   self.__optimizer,
                                                   self.__train_data_loader,
                                                   self.__loss_fn)

            else:
                train_loss = train_one_epoch(self.__model,
                                             self.__optimizer,
                                             self.__train_data_loader,
                                             self.__loss_fn)

            test_loss = self.__test_model()

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

        return train_loss_list, test_loss_list

    def __test_model(self) -> dict[str, float]:

        with torch.no_grad():
            self.__model.eval()

            loss_tensor = torch.zeros(size=(1, len(self.__test_data_loader)), dtype=torch.float64)
            eq_tensor = torch.zeros(size=(1, len(self.__test_data_loader)), dtype=torch.float64)

            for idx, batch in enumerate(self.__test_data_loader):
                feature_tensor, target_tensor = batch

                output = self.__model(feature_tensor)

                loss = self.__loss_fn(output.type(torch.DoubleTensor), target_tensor)

                eq_count = torch.sum(torch.eq(torch.round(output), target_tensor))

                loss_tensor[0][idx] = loss.item()
                eq_tensor[0][idx] = eq_count.item() / target_tensor.size()[0]

            return {"test_loss": torch.mean(loss_tensor).item(), "test_acc": torch.mean(eq_tensor).item()}

    def validate_model(self) -> float:
        with torch.no_grad():
            self.__model.eval()
            feature_tensor = torch.tensor(self.__val_feature_tensor,
                                          device=self.__device).type(torch.DoubleTensor)
            target_tensor = torch.tensor(self.__val_target_tensor,
                                         device=self.__device).type(torch.DoubleTensor)

            output = self.__model(feature_tensor)

            eq_count = torch.sum(torch.eq(torch.round(output), target_tensor))

            return (eq_count / len(self.__val_feature_tensor)).item()
