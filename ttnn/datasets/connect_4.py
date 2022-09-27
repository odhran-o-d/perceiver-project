from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from ttnn.datasets.base import BaseDataset
from ttnn.utils.data_loading_utils import download


class Connect4Dataset(BaseDataset):
    def __init__(self, c):
        super(Connect4Dataset, self).__init__(
            is_imputation=False, fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Binary classification.
        Target in last column. (Theoretical outcome of the game in each state.)
        67557 rows.
        Each row has 42 features.
        All features are categorical and describe the state of the board.

        Class imbalance is [6449, 16635, 44473] which gives a guessing
        accuracy of 0.658.

        """
        path = Path(self.c.data_path) / self.c.data_set
        data_name = 'connect-4.data'
        file = path / data_name

        if not file.is_file():
            # download if does not exist
            download_name = 'connect-4.data.Z'
            url = (
                    'https://archive.ics.uci.edu/ml/'
                    + 'machine-learning-databases/connect-4/'
                    + download_name
            )
            download_file = path / download_name
            download(download_file, url)
            # Connect-4 comes compressed.
            patoolib.extract_archive(str(download_file), outdir=str(path))

        self.data_table = pd.read_csv(file, header=None).to_numpy()

        if self.c.exp_smoke_test:
            print('Running smoke test -- building simple connect-4 dataset.')
            d_win = self.data_table[self.data_table[:, -1] == 'win']
            d_loss = self.data_table[self.data_table[:, -1] == 'loss']
            d_draw = self.data_table[self.data_table[:, -1] == 'draw']
            self.data_table = np.concatenate([d_win, d_loss, d_draw], 0)

        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]
        self.cat_features = list(range(self.D))
        self.num_features = []

        # Target col is the last feature (binary class)
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['connect-4.data', 'connect-4.data.Z']
