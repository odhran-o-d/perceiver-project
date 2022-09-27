from os import remove
from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from ttnn.datasets.base import BaseDataset
from ttnn.utils.data_loading_utils import download


class CiteSeerDataset(BaseDataset):
    """
    The CiteSeer dataset consists of 3312 scientific publications
    classified into one of six classes.
    The citation network consists of 4732 links.

    Each publication in the dataset is described by a 0/1-valued word vector
     indicating the absence/presence of the corresponding word from the
     dictionary. The dictionary consists of 3703 unique words.

    The README file in the dataset provides more details.
    """
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix) = load_and_preprocess_citeseer_dataset(
            self.c)

        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]
        self.is_data_loaded = True
        self.tmp_file_names = []


def load_and_preprocess_citeseer_dataset(c):
    path = Path(c.data_path)
    data_name = 'citeseer.content'
    file = path / c.data_set / data_name

    # For breast cancer, target index is the first column
    if not file.is_file():
        # download if does not exist
        download_name = 'citeseer.tgz'
        url = 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz'
        download_file = path / download_name
        download(download_file, url)

        # Cora comes compressed.
        print('Decompressing...')
        patoolib.extract_archive(str(download_file), outdir=str(path))
        print('... done.')

        remove(download_file)
        print(f'Removed compressed file {download_name}.')

    data_table = pd.read_csv(file, header=None, delimiter='\t').to_numpy()

    # Remove string ID feature
    data_table = data_table[:, 1:]

    N, D = data_table.shape
    num_features = []
    cat_features = list(range(D))
    missing_matrix = np.zeros((N, D), dtype=np.bool_)

    return data_table, N, D, cat_features, num_features, missing_matrix
