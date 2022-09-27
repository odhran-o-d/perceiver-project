from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import patoolib

from ttnn.datasets.base import BaseDataset
from ttnn.utils.data_loading_utils import download


class SarcosDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=-4449)
        self.c = c

    def load(self):
        """Sarcos Dataset.

        Used in TabNet / Adaptive Neural Trees.

        Regression.

        Cite from: http://www.gaussianprocess.org/gpml/data/:
        The data relates to an inverse dynamics problem for a seven
        degrees-of-freedom SARCOS anthropomorphic robot arm. The task is to
        map from a 21-dimensional input space (7 joint positions, 7 joint
        velocities, 7 joint accelerations) to the corresponding 7 joint
        torques. Following previous work we present results for just one of the
        seven mappings, from the 21 input variables to the first of the seven
        torques.

        sarcos_inv        44484x28                  9964416  double array
        sarcos_inv_test    4449x28                   996576  double array

        There are 44,484 training examples and 4,449 test examples.
        The first 21 columns are the input variables, and the 22nd column is
        used as the target variable.

        Note: So really, nobody is doing the full 7 dimensional regression?
        From the appendix of adaptive neural trees: it does seem like they do
        the full 7 targets. Since Tabnet error is pretty similar, it does
        seem like they also predict for full 7 targets.
        """
        path = Path(self.c.data_path) / self.c.data_set

        data_names = ['sarcos_inv.mat', 'sarcos_inv_test.mat']
        files = [path / data_name for data_name in data_names]

        files_exist = [file.is_file() for file in files]

        if not all(files_exist):
            url = 'http://www.gaussianprocess.org/gpml/data/'

            urls = [url + data_name for data_name in data_names]

            download(files, urls)

        data_tables = [loadmat(file) for file in files]
        data_tables = [
            table[name.split('.')[0]] for table, name
            in zip(data_tables, data_names)]
        self.data_table = np.concatenate(data_tables, 0)
        self.N, self.D = self.data_table.shape

        # Last seven columns are numerical targets
        self.num_target_cols = list(range(self.D - 7, self.D))
        self.cat_target_cols = []

        self.cat_features = []
        self.num_features = list(range(0, self.D))
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = data_names
