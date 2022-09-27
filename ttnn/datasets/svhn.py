from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
from sklearn.datasets import fetch_openml

from ttnn.datasets.base import BaseDataset


class SVHNDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=-26032)
        self.c = c

    def load(self):
        # Load data from https://www.openml.org/d/41081
        data_home = Path(self.c.data_path) / self.c.data_set

        svhn_train = datasets.SVHN(
            root=data_home, download=self.c.data_force_reload, split='train')
        svhn_test = datasets.SVHN(
            root=data_home, download=self.c.data_force_reload, split='test')
        print('Fetched SVHN train and test datasets.')

        train_data = torch.flatten(
            torch.Tensor(svhn_train.data), start_dim=1).numpy()
        print(f'Train features shape: {train_data.shape}')
        train_labels = svhn_train.labels

        test_data = torch.flatten(
            torch.Tensor(svhn_test.data), start_dim=1).numpy()
        print(f'Test features shape: {test_data.shape}')
        test_labels = svhn_test.labels

        train_table = np.hstack(
            (train_data, np.expand_dims(train_labels, -1)))
        test_table = np.hstack(
            (test_data, np.expand_dims(test_labels, -1)))

        self.data_table = np.vstack((train_table, test_table))

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last column
        self.cat_target_cols = [self.D - 1]
        self.num_target_cols = []

        self.num_features = list(range(self.D - 1))
        self.cat_features = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['train_32x32.mat', 'test_32x32.mat']
