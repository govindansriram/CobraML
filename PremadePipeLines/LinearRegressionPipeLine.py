import math

import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from Regression.LinearModel import LinearRegression


def get_loss(output: torch.DoubleTensor,
             target: torch.DoubleTensor) -> dict[str, torch.tensor]:
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    mse_loss = loss_mse(output, target)

    return {"mse": mse_loss,
            "mae": loss_mae(output, target),
            "rmse": torch.sqrt(mse_loss)}


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


class LinearRegressionModel:

    def __init__(self,
                 train_feature_tensor: torch.Tensor,
                 train_target_tensor: torch.Tensor,
                 test_feature_tensor: torch.Tensor,
                 test_target_tensor: torch.Tensor,
                 batch_size: int,
                 learning_rate,
                 momentum,
                 randomize=False,
                 device_name="cpu",
                 optimizer="sgd"):

        if device_name != "cpu" and device_name != "cuda":
            raise Exception("not a valid device")

        self.__device = device_name

        if len(train_feature_tensor.size()) > 2 or len(train_target_tensor.size()) < 2:
            raise Exception("tensor shape should be 2-d")

        if len(test_feature_tensor.size()) > 2 or len(test_target_tensor.size()) < 2:
            raise Exception("tensor shape should be 2-d")

        feature_tensor_train = train_feature_tensor.type(torch.DoubleTensor)
        target_tensor_train = train_target_tensor.type(torch.DoubleTensor)

        feature_tensor_test = test_feature_tensor.type(torch.DoubleTensor)
        target_tensor_test = test_target_tensor.type(torch.DoubleTensor)

        dataset_train = LinearRegressionDataset(feature_tensor_train,
                                                target_tensor_train)

        dataset_test = LinearRegressionDataset(feature_tensor_test,
                                               target_tensor_test)

        self.__dataloader_train = DataLoader(dataset_train,
                                             batch_size=batch_size,
                                             shuffle=True)

        self.__dataloader_test = DataLoader(dataset_test,
                                            batch_size=batch_size,
                                            shuffle=True)

        self.__model = LinearRegression(feature_tensor_train.size()[1],
                                        device=device_name,
                                        randomize=randomize).to(self.__device)

        self.__optimizer = optim.SGD([self.__model.get_parameters()],
                                     lr=learning_rate,
                                     momentum=momentum
                                     ) if optimizer == "sgd" else optim.Adam([self.__model.get_parameters()]
                                                                             , lr=learning_rate)

    def predict(self, x_features: torch.Tensor):
        x_features = x_features.type(torch.DoubleTensor)
        x_features = x_features.to(self.__device)

        self.__model.eval()
        with torch.no_grad():
            return self.__model(x_features)

    def train_model(self,
                    epochs: int) -> dict[str, torch.tensor]:

        self.__model.train()

        ret_loss_dict = {
            "mse_tensor": torch.zeros(epochs, dtype=torch.float64, device=self.__device),
            "mae_tensor": torch.zeros(epochs, dtype=torch.float64, device=self.__device),
            "rmse_tensor": torch.zeros(epochs, dtype=torch.float64, device=self.__device)
        }

        loss_arr = []
        # loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs),
                          desc="Training Linear Regression Model per epoch",
                          unit="epochs",
                          colour="green"):

            mse_loss = 0
            mae_loss = 0
            rmse_loss = 0

            for batch in tqdm(self.__dataloader_train,
                              desc="Training Linear Regression Model per batch",
                              unit="batch",
                              colour="blue"):
                self.__optimizer.zero_grad()

                x_input, y_target = batch
                x_input = x_input.to(self.__device)
                y_target = y_target.to(self.__device)

                output = self.__model(x_input)

                # loss = loss_fn(output, y_target)

                loss_dict = get_loss(output, y_target)

                loss_dict["mse"].backward()

                self.__optimizer.step()

                mse_loss += loss_dict["mse"].item()
                mae_loss += loss_dict["mae"].item()
                rmse_loss += loss_dict["rmse"].item()

            ret_loss_dict["rmse_tensor"][epoch] = rmse_loss / len(self.__dataloader_train)
            ret_loss_dict["mae_tensor"][epoch] = mae_loss / len(self.__dataloader_train)
            ret_loss_dict["mse_tensor"][epoch] = mse_loss / len(self.__dataloader_train)

        return ret_loss_dict

    def test_model(self) -> dict[str, torch.tensor]:
        self.__model.eval()

        ret_loss_dict = {
            "mse_loss": 0,
            "mae_loss": 0,
            "rmse_loss": 0
        }

        with torch.no_grad():
            for batch in tqdm(self.__dataloader_test,
                              desc="Testing Linear Regression Model per batch",
                              unit="batch",
                              colour="blue"):
                x_input, y_target = batch
                x_input = x_input.to(self.__device)
                y_target = y_target.to(self.__device)

                output = self.__model(x_input)

                ret_loss_dict["mae_loss"] += torch.sum(torch.abs(output - y_target)).item()
                ret_loss_dict["mse_loss"] += torch.sum(torch.square(output - y_target)).item()

            ret_loss_dict["mae_loss"] /= (ret_loss_dict["mae_loss"] / (len(self.__dataloader_test) * output.size()[0]))
            ret_loss_dict["mse_loss"] /= (ret_loss_dict["mse_loss"] / (len(self.__dataloader_test) * output.size()[0]))
            ret_loss_dict["rmse_loss"] += math.sqrt(ret_loss_dict["mse_loss"])

            #     loss_dict = get_loss(output, y_target)
            #
            #     ret_loss_dict["mse_loss"] += loss_dict["mse"].item()
            #     ret_loss_dict["mae_loss"] += loss_dict["mae"].item()
            #     ret_loss_dict["rmse_loss"] += loss_dict["rmse"].item()
            #
            # ret_loss_dict["mse_loss"] /= len(self.__dataloader_test)
            # ret_loss_dict["mae_loss"] /= len(self.__dataloader_test)
            # ret_loss_dict["rmse_loss"] /= len(self.__dataloader_test)

            return ret_loss_dict
