from typing import Union
import torch
from torch.utils.data import DataLoader


def train_one_epoch_lbfgs(model: torch.nn.Module,
                          optimizer: torch.optim.LBFGS,
                          train_data_loader: DataLoader,
                          loss_fn: Union[torch.nn.MSELoss, torch.nn.BCELoss],
                          device: torch.device) -> torch.Tensor:
    model.train()

    loss_tensor = torch.zeros(size=[1, len(train_data_loader)], dtype=torch.float64, device=device)

    for idx, batch in enumerate(train_data_loader):
        feature_tensor, target_tensor = batch

        feature_tensor = feature_tensor.to(device)
        target_tensor = target_tensor.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(feature_tensor)
            loss = loss_fn(output.float(), target_tensor.float())
            loss.backward()
            return loss

        optimizer.step(closure)

        run_loss = closure()
        loss_tensor[0][idx] += run_loss

    return torch.mean(loss_tensor)


def train_one_epoch(model: torch.nn.Module,
                    optimizer: Union[torch.optim.Adam, torch.optim.SGD],
                    train_data_loader: DataLoader,
                    loss_fn: Union[torch.nn.MSELoss, torch.nn.BCELoss],
                    device: torch.device) -> torch.Tensor:
    model.train()

    loss_tensor = torch.zeros(size=[1, len(train_data_loader)],
                              dtype=torch.float64,
                              device=device)

    for idx, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        feature_tensor, target_tensor = batch

        feature_tensor = feature_tensor.to(device)
        target_tensor = target_tensor.to(device)

        output = model(feature_tensor)

        loss = loss_fn(output.float().to(device), target_tensor.float())

        loss.backward()

        optimizer.step()

        loss_tensor[0][idx] += loss

    return torch.mean(loss_tensor)
