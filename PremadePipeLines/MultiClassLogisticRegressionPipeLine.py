import random

import torch
from PremadePipeLines.BinaryLogisticRegressionPipeLine import BinaryLogisticRegression


class MultiClassLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[list[int]],
                 batch_size: int,
                 learning_rate: float,
                 momentum: float,
                 epochs: int,
                 train_split_multi=0.90,
                 train_split_bin=0.80,
                 randomize=False,
                 device_name="cpu",
                 optimizer="SGD"):

        optim_list = ("SGD",
                      "ADAM",
                      "LBFGS")

        if optimizer not in optim_list:
            raise Exception("Invalid optimizer")

        if train_split_bin >= 1 or train_split_multi >= 1:
            raise Exception("split exceeds 100%")

        self.__device = torch.device(device_name)

        target_tensor = torch.tensor(target_data)

        unique = torch.unique(torch.unsqueeze(target_tensor, dim=1), dtype=torch.int64)

        data = list(zip(feature_data, target_data))
        random.shuffle(data)

        feature_list, target_list = list(zip(*data))
        feature_list = list(feature_list)
        target_list = list(target_list)

        cut_off_num = round(len(feature_list) * train_split_bin)

        feature_list_bin = feature_list[0: cut_off_num]

        target_tensor_bin = torch.tensor(target_list[0: cut_off_num], dtype=torch.int64, device=self.__device)

        self.__feature_list_mult = torch.tensor(feature_list[cut_off_num: len(feature_list)],
                                                device=self.__device,
                                                dtype=torch.float64)

        self.__target_list_mult = torch.tensor(target_list[cut_off_num: len(feature_list)],
                                               device=self.__device,
                                               dtype=torch.int64)

        model_dict_list = []

        for value in unique:

            for i in range(target_tensor_bin.size()[0]):
                target_tensor_bin[i][0] = 1 if target_tensor_bin[i][0] == value else 0

            current_model = BinaryLogisticRegression(feature_list_bin,
                                                     target_tensor_bin.tolist(),
                                                     batch_size,
                                                     learning_rate,
                                                     momentum,
                                                     train_split_bin,
                                                     randomize,
                                                     device_name=device_name,
                                                     optimizer=optimizer)

            current_model.fit_model(epochs)

            model_dict_list.append({"model": current_model,
                                    "class": value})

        model_dict_list.sort(key=lambda x: x["class"])

    def validate_model(self) -> torch.Tensor:
        with torch.no_grad():
            self.__model.eval()
            feature_tensor = torch.tensor(self.__val_feature_tensor,
                                          device=self.__device).float()
            target_tensor = torch.tensor(self.__val_target_tensor,
                                         device=self.__device).float()

            output = self.__model(feature_tensor.to(self.__device))

            eq_count = torch.sum(torch.eq(torch.round(output), target_tensor.to(self.__device)))

            return eq_count / len(self.__val_feature_tensor)
