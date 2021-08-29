from typing import Union
import torch
from torch.utils.data import DataLoader


def train_one_epoch_lbfgs(model: torch.nn.Module,
                          optimizer: torch.optim.LBFGS,
                          train_data_loader: DataLoader,
                          loss_fn) -> float:
    model.train()

    loss_tensor = torch.zeros(size=[1, len(train_data_loader)], dtype=torch.float64)

    for idx, batch in enumerate(train_data_loader):
        feature_tensor, target_tensor = batch

        def closure():
            optimizer.zero_grad()
            output = model(feature_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        run_loss = closure()
        loss_tensor[0][idx] += run_loss.item()

    return torch.mean(loss_tensor).item()


def train_one_epoch(model: torch.nn.Module,
                    optimizer: Union[torch.optim.Adam, torch.optim.SGD],
                    train_data_loader: DataLoader,
                    loss_fn) -> float:
    model.train()

    loss_tensor = torch.zeros(size=[1, len(train_data_loader)], dtype=torch.float64)

    for idx, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        feature_tensor, target_tensor = batch

        output = model(feature_tensor)

        loss = loss_fn(output.type(torch.DoubleTensor), target_tensor)

        loss.backward()

        optimizer.step()

        loss_tensor[0][idx] += loss.item()

    return torch.mean(loss_tensor).item()
