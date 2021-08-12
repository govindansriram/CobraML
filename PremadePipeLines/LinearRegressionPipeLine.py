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
                 train_feature_tensor: torch.Tensor,
                 train_target_tensor: torch.Tensor,
                 test_feature_tensor: torch.Tensor,
                 test_target_tensor: torch.Tensor,
                 batch_size: int,
                 learning_rate,
                 momentum,
                 randomize=False,
                 device_name="cpu"):

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
                                     momentum=momentum)

    def predict(self, x_features: torch.Tensor):
        x_features = x_features.type(torch.DoubleTensor)
        x_features = x_features.to(self.__device)

        self.__model.eval()
        # self.__model.get_parameters().eval()

        with torch.no_grad():
            return self.__model(x_features)

    def train_model(self,
                    epochs):

        self.__model.train()
        # self.__model.get_parameters().train()

        loss_arr = []
        loss_fn = nn.MSELoss()

        for _ in tqdm(range(epochs),
                      desc="Training Linear Regression Model per epoch",
                      unit="epochs",
                      colour="green"):

            epoch_loss = 0

            for batch in tqdm(self.__dataloader_train,
                              desc="Training Linear Regression Model per batch",
                              unit="batch",
                              colour="blue"):
                self.__optimizer.zero_grad()

                x_input, y_target = batch
                x_input = x_input.to(self.__device)
                y_target = y_target.to(self.__device)

                output = self.__model(x_input)

                loss = loss_fn(output, y_target)

                loss.backward()

                self.__optimizer.step()

                epoch_loss += loss.item()

            loss_arr.append(epoch_loss / len(self.__dataloader_train))

        return loss_arr

    def test_model(self):
        self.__model.eval()
        # self.__model.get_parameters().eval()

        diff_val = 0

        with torch.no_grad():
            for batch in tqdm(self.__dataloader_test,
                              desc="Testing Linear Regression Model per batch",
                              unit="batch",
                              colour="blue"):

                x_input, y_target = batch
                x_input = x_input.to(self.__device)
                y_target = y_target.to(self.__device)

                output = self.__model(x_input)

                abs_tensor = torch.abs(y_target - output)

                diff_val += torch.sum(abs_tensor).item

            return diff_val / len(self.__dataloader_test)
