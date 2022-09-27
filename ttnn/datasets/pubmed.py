from os import remove
from pathlib import Path

import numpy as np
import pandas as pd
import patoolib

from ttnn.datasets.base import BaseDataset
from ttnn.utils.data_loading_utils import download


class PubMedDataset(BaseDataset):
    """
    The Pubmed Diabetes dataset consists of 19717 scientific publications
     from PubMed database pertaining to diabetes classified into
      one of three classes.

      The citation network consists of 44338 links.

      Each publication in the dataset is described by a TF/IDF weighted word
       vector from a dictionary which consists of 500 unique words.

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
    path = Path(c.data_path) / c.data_set
    data_name = (
            Path('Pubmed-Diabetes') / 'data' / 'Pubmed-Diabetes.NODE.paper.tab')
    file = path / data_name

    # For breast cancer, target index is the first column
    if not file.is_file():
        # download if does not exist
        download_name = 'pubmed.tgz'
        url = 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz'
        download_file = path / download_name
        download(download_file, url)

        # Cora comes compressed.
        print('Decompressing...')
        patoolib.extract_archive(str(download_file), outdir=str(path))
        print('... done.')

        remove(download_file)
        print(f'Removed compressed file {download_name}.')

    # Each column header specifies if the variable is categorical or numerical
    data_table = pd.read_csv(
        file, index_col=0, skiprows=[0], delimiter='\t', engine='python')

    # Drop the summary column, which only summarizes which other columns
    # are non-zero for each row
    data_table.drop(columns={'string:summary'}, inplace=True)

    def convert_element(elem):
        if not pd.isna(elem):
            elem = elem.split('=')[1]

        return elem

    data_table = data_table.apply(np.vectorize(convert_element))

    # Convert each row to numerical - if it fails, just drop the row
    rows_to_delete = []
    for row_index in range(data_table.shape[0]):
        try:
            data_table.iloc[row_index] = pd.to_numeric(data_table.iloc[row_index])
        except:
            rows_to_delete.append(data_table.iloc[row_index].index)

    print(rows_to_delete)

    print(
        f'Deleted {rows_to_delete} rows, which were incorrectly '
        f'formatted in PubMed.')

    # TODO: fix - not working

    for column in data_table.columns:
        data_table[column] = pd.to_numeric(data_table[column])

    data_table = data_table.to_numpy()

    # Remove string ID feature
    data_table = data_table[:, 1:]

    print(data_table)

    N, D = data_table.shape
    num_features = list(range(1, D))
    cat_features = list(range(D))
    missing_matrix = np.zeros((N, D), dtype=np.bool_)

    return data_table, N, D, cat_features, num_features, missing_matrix
