import torch


def squared_error(pred_tensor, output_tensor):
    return torch.sum(torch.square(torch.subtract(pred_tensor, output_tensor))).item() * (1 / (2 * len(pred_tensor)))
