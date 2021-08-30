import torch
from PremadePipeLines.BinaryLogisticRegressionPipeLine import BinaryLogisticRegression


class MultiClassLogisticRegression:

    def __init__(self,
                 feature_data: list[list[float]],
                 target_data: list[list[int]],
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 device_name="cpu"):

        self.__device = torch.device(device_name)

        target_tensor = torch.tensor(target_data,
                                     device=self.__device,
                                     dtype=torch.float64)

        unique = torch.unique(torch.unsqueeze(target_tensor, dim=1))

        unique = unique.type(torch.IntTensor)

        self.__model_dict_list = []

        for value in unique:

            for i in range(target_tensor.size()[0]):
                target_tensor[i][0] = 1 if target_tensor[i][0] == value else 0

            current_model = BinaryLogisticRegression(feature_data,
                                                     target_tensor.tolist(),
                                                     batch_size,
                                                     learning_rate,
                                                     device_name=device_name)

            loss = current_model.fit_model(epochs)

            self.__model_dict_list.append({"model": current_model,
                                           "class": value})

        self.__model_dict_list.sort(key=lambda x: x["class"])

    def validate_model(self,
                       feature_data: list[list[float]],
                       target_data: list[list[int]], ) -> torch.Tensor:

        with torch.no_grad():
            feature_tensor = torch.tensor(feature_data, dtype=torch.float64, device=self.__device)
            target_tensor = torch.tensor(target_data, dtype=torch.float64, device=self.__device)
            output = self.__model_dict_list[0]["model"].get_model()(feature_tensor)

            for model_dict in self.__model_dict_list[1:]:
                new_output = model_dict["model"].get_model()(feature_tensor)
                output = torch.cat((output, new_output), dim=1)

            index_tensor = torch.unsqueeze(torch.max(output, dim=1)[1], dim=1)
            eq_count = torch.sum(torch.eq(index_tensor, target_tensor))

            print(index_tensor[:10, :])
            print('\n')
            print(output[:10, :])
            print('\n')
            print(target_tensor[:10, :])

            return eq_count / target_tensor.size()[0]
