import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Regression.LinearModel import LinearRegression


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
                 feature_tensor: torch.Tensor,
                 target_tensor: torch.Tensor,
                 batch_size: int,
                 randomize=False,
                 device_name="cpu"):

        if device_name != "cpu" and device_name != "cuda":
            raise Exception("not a valid device")

        self.__device = device_name

        if len(feature_tensor.size()) > 2 or len(feature_tensor.size()) < 2:
            raise Exception("tensor shape should be 2-d")

        feature_tensor, target_tensor = feature_tensor.type(torch.DoubleTensor), target_tensor.type(torch.DoubleTensor)

        dataset = LinearRegressionDataset(feature_tensor,
                                          target_tensor)

        self.__dataloader = DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True)

        self.__model = LinearRegression(feature_tensor.size()[1],
                                        device=device_name,
                                        randomize=randomize).to(self.__device)

    def predict(self, x_features: torch.Tensor):
        x_features = x_features.type(torch.DoubleTensor)
        x_features = x_features.to(self.__device)

        self.__model.eval()
        return self.__model(x_features)

    def train_model(self,
                    epochs,
                    learning_rate,
                    momentum,
                    optim_name="sgd"):

        loss_arr = []

        if optim_name == "sgd":
            optimizer = optim.SGD([self.__model.get_parameters()],
                                  lr=learning_rate,
                                  momentum=momentum)
        else:
            raise Exception("not a valid optimizer")

        loss_fn = nn.MSELoss()

        for _ in tqdm(range(epochs),
                      desc="Training Linear Regression Model per epoch",
                      unit="epochs",
                      colour="green"):

            epoch_loss = 0

            for batch in tqdm(self.__dataloader,
                              desc="Training Linear Regression Model per batch",
                              unit="batch",
                              colour="blue"):
                optimizer.zero_grad()

                x_input, y_target = batch
                x_input = x_input.to(self.__device)
                y_target = y_target.to(self.__device)

                output = self.__model(x_input)

                loss = loss_fn(output, y_target)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            loss_arr.append(epoch_loss / len(self.__dataloader))

        return loss_arr
