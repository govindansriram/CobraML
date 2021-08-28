import torch


def add_extra_input(x_input: torch.FloatTensor,
                    device) -> torch.Tensor:
    ones_arr = torch.ones((x_input.shape[0], 1),
                          dtype=torch.float64,
                          device=device)

    return torch.cat((ones_arr, x_input),
                     dim=1)
