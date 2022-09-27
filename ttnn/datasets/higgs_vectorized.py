import os

import numpy as np
import pandas as pd


def prepare_higgs_dataset(c):
    # For Higgs, target index is the first column
    target_col = 0

    data_path = os.path.join(c.data_path, c.data_set, 'data.csv')

    # Read dataset in chunks
    for df_chunk in pd.read_csv(
            data_path, chunksize=c.data_chunk_size, header=None):
        data_table = df_chunk.to_numpy()
        N = data_table.shape[0]
        D = data_table.shape[1]

        if c.exp_smoke_test:
            print('Running smoke test -- building simple Higgs dataset.')
            dm = data_table[data_table[:, 0] == 1.0][:8, :5]
            db = data_table[data_table[:, 0] == 0.0][:8, :5]
            data_table = np.concatenate([dm, db], 0)
            N = data_table.shape[0]
            D = data_table.shape[1]

            # Speculate some spurious missing features
            missing_matrix = np.zeros((N, D))
            missing_matrix[0, 1] = 1
            missing_matrix[2, 2] = 1
            missing_matrix = missing_matrix.astype(dtype=np.bool_)
        else:
            missing_matrix = np.zeros((N, D))
            missing_matrix[0, 1] = 1
            missing_matrix[2, 2] = 1
            missing_matrix = missing_matrix.astype(dtype=np.bool_)

        cat_features = [0]
        num_features = list(range(1, D))
        yield {
            'data_table': data_table,
            'missing_matrix': missing_matrix,
            # TODO: update for num_target_cols, cat_target_cols vars
            # 'target_col': target_col,
            'N': N,
            'D': D,
            'cat_features': cat_features,
            'num_features': num_features,
            'is_imputation': False
            }
