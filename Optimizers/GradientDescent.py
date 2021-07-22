import torch


def batch_grad(learning_rate, inputs, targets, predictions, parameters):
    updated_param = []

    diff_tensor = torch.subtract(predictions, targets)
    for idx, theta in enumerate(parameters):
        mul_tensor = torch.matmul(diff_tensor, inputs[:, idx:idx + 1])
        updated_param.append(theta - (learning_rate * torch.sum(mul_tensor).item()))

    return torch.FloatTensor(updated_param)
