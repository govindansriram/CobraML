import torch
from torch import nn, optim
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

            data_set = LogisticRegressionDataset(torch.tensor(train_feature_tensor,
                                                              device=self.__device).type(torch.DoubleTensor),
                                                 torch.tensor(train_target_tensor,
                                                              device=self.__device).type(torch.DoubleTensor))

            train_data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

            test_target_tensor = list(map(lambda x: 1 if x == curr_class else 0, test_target_tensor))

            data_set = LogisticRegressionDataset(torch.tensor(test_feature_tensor,
                                                              device=self.__device).type(torch.DoubleTensor),
                                                 torch.tensor(test_target_tensor,
                                                              device=self.__device).type(torch.DoubleTensor))

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

    def fit_model(self, epochs) -> dict[str: list[dict[str: float]]]:

        model_loss_dict = {}
        for model_dict in self.__class_dict:
            loss_list = []

            for epoch in range(epochs):
                if isinstance(model_dict["optimizer"], optim.lbfgs.LBFGS):
                    train_loss = self.__train_one_epoch_lbfgs(model_dict["model"],
                                                              model_dict["optimizer"],
                                                              model_dict["train_data_loader"])

                else:
                    train_loss = self.__train_one_epoch(model_dict["model"],
                                                        model_dict["optimizer"],
                                                        model_dict["train_data_loader"])

                test_loss = self.__test_model(model_dict["model"],
                                              test_data_loader=model_dict["test_data_loader"])

                loss_dict = {"epoch": float(epoch),
                             "train_loss": train_loss,
                             "test_loss": test_loss[0],
                             "test_acc": test_loss[1]}

                loss_list.append(loss_dict)

                model_loss_dict["class_name"] = {"loss_list": loss_list}

            return model_loss_dict

    def __train_one_epoch(self,
                          model: LogisticRegression,
                          optimizer: optim,
                          train_data_loader: DataLoader) -> float:

        model.train()

        loss_tensor = torch.zeros(size=[1, len(train_data_loader)], dtype=torch.float64)

        for idx, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            feature_tensor, target_tensor = batch

            output = model(feature_tensor)

            loss = self.__loss_fn(output.type(torch.DoubleTensor), target_tensor)

            loss.backward()

            optimizer.step()

            loss_tensor[0][idx] += loss.item()

        return torch.mean(loss_tensor).item()

    def __train_one_epoch_lbfgs(self,
                                model: LogisticRegression,
                                optimizer: optim.lbfgs.LBFGS,
                                train_data_loader: DataLoader) -> float:

        model.train()

        loss_tensor = torch.zeros(size=[1, len(train_data_loader)], dtype=torch.float64)

        for idx, batch in enumerate(train_data_loader):
            feature_tensor, target_tensor = batch

            def closure():
                optimizer.zero_grad()
                output = model(feature_tensor)
                loss = self.__loss_fn(output, target_tensor)
                loss.backward()
                return loss

            optimizer.step(closure)

            end_output = model(feature_tensor)
            loss_tensor[0][idx] += self.__loss_fn(end_output, target_tensor).item()

        return torch.mean(loss_tensor).item()

    def __test_model(self,
                     model: LogisticRegression,
                     test_data_loader: DataLoader) -> tuple[float, float]:

        with torch.no_grad():
            model.eval()

            loss_tensor = torch.zeros(size=(1, len(test_data_loader)), dtype=torch.float64)
            eq_tensor = torch.zeros(size=(1, len(test_data_loader)), dtype=torch.float64)

            for idx, batch in enumerate(test_data_loader):
                feature_tensor, target_tensor = batch

                output = model(feature_tensor)

                loss = self.__loss_fn(output.type(torch.DoubleTensor), target_tensor)

                eq_count = torch.sum(torch.eq(torch.round(output), target_tensor))

                loss_tensor[0][idx] = loss.item()
                eq_tensor[0][idx] = eq_count.item() / target_tensor.size()[0]

            return (torch.mean(loss_tensor).item(),
                    torch.mean(eq_tensor).item())

    def _validate_model(self):
        feature_tensor = torch.tensor(self.__val_feature_tensor, device=self.__device).type(torch.DoubleTensor)
        target_tensor = torch.tensor(self.__val_target_tensor, device=self.__device).type(torch.DoubleTensor)

        output = self.__class_dict[0]["model"](feature_tensor)

        for model_dict in self.__class_dict[1:]:
            new_output = model_dict["model"](feature_tensor)
            output = torch.cat((output, new_output), dim=1)

        eq_count = torch.eq(torch.unsqueeze(torch.max(output, 1)[1], dim=1).type(torch.DoubleTensor),
                            target_tensor)

        return eq_count / len(self.__val_feature_tensor)
