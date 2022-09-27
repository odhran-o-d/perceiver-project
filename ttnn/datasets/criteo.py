import os
from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from ttnn.datasets.base import BaseDataset


# TODO: return training and testing data as intended
# TODO: decide on fixed dataset size
class CriteoDataset(BaseDataset):
    def __init__(self, c):
        super(CriteoDataset, self).__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """Criteo dataset as used by AutoINT.

        Original download unavailable. We found alternative host.

        Data Description from www.kaggle.com/c/criteo-display-ad-challenge/data
        Label - Target variable, indicates if an ad was clicked (1) or not (0).
        I1-I13 - A total of 13 columns of int features (mostly count features).
        C1-C26 - A total of 26 columns of categorical features. The values of
            these features have been hashed onto 32 bits for anonymization
            purposes.

        Original files unavailable, we are using this host.
        https://figshare.com/articles/
        Kaggle_Display_Advertising_Challenge_dataset/5732310

        Cannot use test data because it has no labels.

        # TODO: criteo has count columns. model them as categorical?
        """
        path = Path(self.c.data_path) / self.c.data_set
        data_name = 'train.txt'
        file = path / data_name

        if not file.is_file():
            print('Downloading...')
            download_file = '10082655'
            os.system(
                f'wget https://ndownloader.figshare.com/files/{download_file}')

            renamed = 'criteo.tar.gz'
            (Path(download_file)).rename(path / renamed)
            print('Decompressing...')
            patoolib.extract_archive(str(path / renamed), outdir=str(path))
            print('... done.')

        print('Loading data from file...')
        self.data_table = pd.read_csv(file, header=None, sep='\t').to_numpy()
        print('...done.')
        # data_table.shape = (45840617, 40)

        self.N, self.D = self.data_table.shape
        self.num_target_cols = []
        self.cat_target_cols = [0]  # Binary classification
        self.cat_features = [0] + list(range(14, self.D))
        self.num_features = list(range(1, 14))

        # TODO: Missing data? (Looking at the train.txt it seems like there is
        # missing data there!)
        # TODO: they get read as nan by pandas. map nans to missing matrix!
        # TODO: check this code, but it should work
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        idxs = np.where(np.isnan == self.data_table)
        self.missing_matrix[idxs] = True
        print(f'Found {len(idxs[0])} missing values.')
        assert len(idxs[0]) == (self.missing_matrix == True).sum()

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = [
            'criteo.tar.gz', 'readme.txt', 'test.txt', 'train.txt']
