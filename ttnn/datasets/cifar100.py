from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

from ttnn.datasets.base import BaseDataset
import torch

class CIFAR100Dataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Classification dataset.
        This dataset is just like the CIFAR-10, except it has 100 classes
        containing 600 images each. There are 500 training images and 100
        testing images per class.

        The 100 classes in the CIFAR-100 are grouped into 20 superclasses.
        Each image comes with a "fine" label (the class to which it belongs)
        and a "coarse" label (the superclass to which it belongs).
        """

        self.N = 60000
        self.D = 3073
        self.cat_features = [self.D - 1]
        self.num_features = list(range(0, self.D - 1))

        # Target col is the last feature
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = torch.zeros((self.N, self.D), dtype=torch.bool)
        self.is_data_loaded = True
        # self.tmp_file_or_dir_names = ['openml']

        self.input_feature_dims = [1] * 3072
        self.input_feature_dims += [100]
