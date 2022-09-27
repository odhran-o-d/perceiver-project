from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from ttnn.datasets.base import BaseDataset


class EnergyEfficiencyDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Regression dataset.

        Targets in last two columns.
        768 rows.
        8 attributes.
        2 targets

            Features                      n_unique   encode as
            X1  Relative Compactness        12        CAT
            X2  Surface Area                12        CAT
            X3  Wall Area                    7        CAT
            X4  Roof Area                    4        CAT
            X5  Overall Height               2        CAT
            X6  Orientation                  4        CAT
            X7  Glazing Area                 4        CAT
            X8  Glazing Area Distribution    6        CAT
            y1  Heating Load                38        NUM
            y2  Cooling Load                37        NUM
        """

        # Load data from https://www.openml.org/d/1472
        data_home = Path(self.c.data_path) / self.c.data_set
        data = fetch_openml(
            'energy-efficiency', version=1, data_home=data_home)

        self.data_table = np.hstack(
            (data['data'], np.expand_dims(data['target'], -1)))

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]

        # Target col is the last two features
        self.cat_target_cols = []
        self.num_target_cols = [self.D - 2, self.D - 1]

        self.num_features = list(range(8, 10))
        self.cat_features = list(range(8))

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['openml']
