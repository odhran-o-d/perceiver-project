from pathlib import Path
import numpy as np
import patoolib
from sklearn.datasets import load_svmlight_file

from ttnn.datasets.base import BaseDataset
from ttnn.utils.data_loading_utils import download


class EpsilonDataset(BaseDataset):
    def __init__(self, c):
        super(EpsilonDataset, self).__init__(
            is_imputation=False,
            fixed_test_set_index=-100000)  # Test set: last 100,000 examples
        self.c = c

    def load(self):
        """Epsilon dataset as used by NODE.

        Binary classification.
        Target in first column.
        400,000 train and 100,000 test rows.
        Each row has 2001 columns.
        The first column contains the label value, all other columns contain
        numerical features.
        Separate training and test set.
        # TODO: Missing data?
        """
        path = Path(self.c.data_path) / self.c.data_set
        data_names = ['epsilon_normalized', 'epsilon_normalized.t']
        files = [path / data_name for data_name in data_names]

        files_exist = [file.is_file() for file in files]

        if not all(files_exist):

            download_names = ['epsilon_normalized.bz2',
                              'epsilon_normalized.t.bz2']
            download_files = [path / name for name in download_names]

            url = (
                'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/'
                'datasets/binary/')

            urls = [url + download_name for download_name in download_names]

            download(download_files, urls)

            print('Decompressing...')
            for i, name in enumerate(download_files):
                print(f'\t File {i}/{len(download_files)}: {name}.')
                patoolib.extract_archive(str(name), outdir=str(path))
            print('... done.')

        print('Loading data from file....')
        data_tables = []
        for i, file in enumerate(files):
            dt, labels = load_svmlight_file(str(file))
            dt = dt.todense()
            dt = np.concatenate([labels[:, np.newaxis], dt], 1)
            data_tables.append(dt)

        # currently not using train and test as intended
        self.data_table = np.concatenate(data_tables, 0)
        print('... done.')

        self.N, self.D = self.data_table.shape

        # Binary classification
        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.cat_features = [0]
        self.num_features = list(range(1, self.D))
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = data_names + [
            'epsilon_normalized.bz2', 'epsilon_normalized.t.bz2']
