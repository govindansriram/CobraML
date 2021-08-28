import math
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from Classification.LogisticModel import LogisticRegression


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


class LogisticRegressionModel:

    def __init__(self,
                 train_feature_tensor: list[list],
                 train_target_tensor: list[list],
                 test_feature_tensor: list[list],
                 test_target_tensor: list[list],
                 val_feature_tensor: list[list],
                 val_target_tensor: list[list],
                 batch_size: int,
                 learning_rate: float,
                 momentum: float,
                 randomize=False,
                 device_name="cpu",
                 optimizer="SGD"):

        optim_list = ("SGD",
                      "ADAM",
                      "LBFGS")

        if optimizer not in optim_list:
            raise Exception("Invalid optimizer")

        self.__loss_fn = nn.BCELoss()

        class_list = set(train_target_tensor)

        class_num_list = {}

        self.__class_dict = []

        self.__device = torch.device(device_name)

        for idx, curr_class in enumerate(class_list):

            class_num_list[curr_class] = idx

            train_target_tensor = list(map(lambda x: 1 if x == curr_class else 0, train_target_tensor))
            data_set = LogisticRegressionDataset(torch.tensor(train_feature_tensor).type(torch.DoubleTensor),
                                                 torch.tensor(train_target_tensor).type(torch.DoubleTensor))
            train_data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

            test_target_tensor = list(map(lambda x: 1 if x == curr_class else 0, test_target_tensor))
            data_set = LogisticRegressionDataset(torch.tensor(test_feature_tensor).type(torch.DoubleTensor),
                                                 torch.tensor(test_target_tensor).type(torch.DoubleTensor))
            test_data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

            model = LogisticRegression(len(train_feature_tensor[0]),
                                       device=device_name,
                                       randomize=randomize).to(self.__device)

            if optimizer == "SGD":
                curr_optimizer = optim.SGD([model.get_parameters()],
                                           lr=learning_rate,
                                           momentum=momentum)
            elif optimizer == "ADAM":
                curr_optimizer = optim.Adam([model.get_parameters()],
                                            lr=learning_rate)

            else:
                curr_optimizer = optim.LBFGS([model.get_parameters()],
                                             lr=learning_rate)

            self.__class_dict.append({"class_name": curr_class,
                                      "class_number": idx,
                                      "train_data_loader": train_data_loader,
                                      "test_data_loader": test_data_loader,
                                      "model": model,
                                      "optimizer": curr_optimizer})

        self.__val_target_tensor = list(map(lambda x: class_num_list[x], val_target_tensor))
        self.__val_feature_tensor = val_feature_tensor
