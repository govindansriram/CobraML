import torch
from torch.utils.data import DataLoader


class BinaryGDA:

    def __init__(self,
                 feature_count: int,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 device="cpu"):

        self.__train_loader = train_data_loader
        self.__test_loader = test_data_loader

        self.__feature_count = feature_count

        self.__device = torch.device(device=device)
        self.__mu_matrix = torch.zeros(size=(feature_count, 2),
                                       dtype=torch.float64,
                                       device=self.__device)

        self.__co_variance_matrix = torch.zeros(size=(feature_count, feature_count),
                                                dtype=torch.float64,
                                                device=self.__device)
        self.__fi = 0

    def train(self):
        for batch in self.__train_loader:
            feature_tensor, target_tensor = batch

            feature_tensor = feature_tensor.to(self.__device)
            target_tensor = target_tensor.to(self.__device)

            full_tensor = torch.cat((feature_tensor, target_tensor), dim=1)

            mask = (full_tensor[:, self.__feature_count + 1] == 0)

            zero_matrix = full_tensor[mask][:, :self.__feature_count]
            one_matrix = full_tensor[~mask][:, :self.__feature_count]


if __name__ == '__main__':
    other_test_arr = torch.tensor([[100, 1],
                      [12, 18],
                      [15, 19],
                      [10, 70],
                      [50, 100]])

    test_array = torch.tensor([[0],
                  [0],
                  [1],
                  [1],
                  [0]])

    cat_ten = torch.cat((other_test_arr, test_array), dim=1)

    mask = (cat_ten[:, 2] == 1)

    print(cat_ten[mask][:, :2])

    print(cat_ten[~mask][:, :2])