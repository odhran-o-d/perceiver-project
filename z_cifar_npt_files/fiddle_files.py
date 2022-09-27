from dotmap import DotMap
from ttnn.datasets.base import BaseDataset
from pathlib import Path
from os.path import join, isfile
import numpy as np
import pandas as pd
import torch
import sparse
from torch.utils.data import TensorDataset
import pickle
from sklearn.model_selection import train_test_split

from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random, 
                          coo_matrix,
                          csr_matrix, 
                          vstack)
from tqdm import tqdm

class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 targets:Union[np.ndarray, coo_matrix, csr_matrix], 
                 transform:bool = None):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
        self.transform = transform # Can be removed

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s).to_dense()
  
    
def float_sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch dense tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch


def long_sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch dense tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
        data_batch = data_batch.long()
    else:
        data_batch = torch.LongTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
        targets_batch = targets_batch.long()
    else:
        targets_batch = torch.LongTensor(targets_batch)
    return data_batch, targets_batch





def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_and_preprocess_FIDDLE(c, data_name, task):
    path = Path(c.data_path) / c.data_set

    folder = path / data_name / 'features' / task

    complete_data_path = folder / 'complete_data.npz'
    time_code_path = folder / 'time_code.pkl'
    if not isfile(complete_data_path):
        print('saved data not found - creating saved data...')
        static_data = folder / 's.npz'
        static_feat = sparse.load_npz(static_data)

        temporal_data = folder / 'X.npz'
        time_feat = sparse.load_npz(temporal_data)
        features_per_step = time_feat.shape[2]

        time_feat = time_feat.reshape((time_feat.shape[0], -1))

        label_path = folder = path / data_name / 'population' / f'{task}.csv'
        label = pd.read_csv(label_path).iloc[:, -1:].to_numpy()
        label = sparse.COO.from_numpy(label)

        data = sparse.concatenate([label, static_feat, time_feat], axis=-1)

        static_indices = list(range(0, static_feat.shape[1]+1))
        temporal_indices = list(range(static_feat.shape[1]+1, data.shape[1]))
        temporal_indices = list(chunks(temporal_indices, features_per_step))
        time_code = [static_indices] + temporal_indices
        
        # above might crash if label is not int
        print('label, disc and cont concatenated')
        sparse.save_npz(complete_data_path, data)
        with open(time_code_path, 'wb') as handle:
            pickle.dump(time_code, handle)

    # Read dataset
    data_table = sparse.load_npz(complete_data_path)
    with open(time_code_path, 'rb') as handle:
        time_code = pickle.load(handle)

    N = data_table.shape[0]
    D = data_table.shape[1]

    missing_matrix = np.zeros((N, D))
    missing_matrix = missing_matrix.astype(dtype=np.bool_)

    cat_features = list(range(0, D))
    num_features = []
    return data_table, N, D, cat_features, num_features, missing_matrix, time_code


class ARF4heICUDataset(BaseDataset):
    def __init__(self, c):
        super(ARF4heICUDataset, self).__init__(
            is_imputation=False, fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix, self.time_code) = load_and_preprocess_FIDDLE(
            self.c, data_name='FIDDLE_eicu', task='ARF_4h')

        # For breast cancer, target index is the first column
        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['eIUC_ARF_4h']  # TODO: this needs fixing

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0


class ARF4hMIMICDataset(BaseDataset):
    def __init__(self, c):
        super(ARF4hMIMICDataset, self).__init__(
            is_imputation=False, fixed_test_set_index=None)
        self.c = c

    def load(self):
        (self.data_table, self.N, self.D, self.cat_features, self.num_features,
            self.missing_matrix, self.time_code) = load_and_preprocess_FIDDLE(
            self.c, data_name='FIDDLE_mimic3', task='ARF_4h')

        # self.data_table = self.data_table.todense()
        # For breast cancer, target index is the first column
        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['MIMIC_ARF_4h']  # TODO: this needs fixing

        # overwrite missing
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0


def get_mimic(
    target_in_training = False,
    test_split = 0.2,
    random_state = 42):


    c = {
        'data_path': 'data',
        'data_set': 'mimic',
        'exp_artificial_missing': 0}

    data_obj = ARF4hMIMICDataset(DotMap(c))

    data_obj.load()

    y = data_obj.data_table[:, 0].todense()
    if target_in_training: 
        X = data_obj.data_table.to_scipy_sparse()
        X = X.tocsr()
        X[:, 0] = 2 #set to mask token 
        X = X.tocoo()

    else:
        X = data_obj.data_table[:, 1:].to_scipy_sparse()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    train_dataset = SparseDataset(X_train, y_train)
    test_dataset = SparseDataset(X_test, y_test)
    time_code = data_obj.time_code

    if not target_in_training:
        del(time_code[0][0])  # these are stupid fixes to remove the targets from the 
        for i in range(len(time_code)):
            for j in range(len(time_code[i])):
                time_code[i][j] += -1



    num_classes = 2
    input_size = 16278

    return input_size, num_classes, train_dataset, test_dataset, time_code

def get_dense_mimic(
    target_in_training = False,
    test_split = 0.2,
    random_state = 42,
    return_tensors = False):


    c = {
        'data_path': 'data',
        'data_set': 'mimic',
        'exp_artificial_missing': 0}

    data_obj = ARF4hMIMICDataset(DotMap(c))

    data_obj.load()

    time_code = data_obj.time_code

    if not target_in_training:
        del(time_code[0][0])  # these are stupid fixes to remove the targets from the 
        for i in range(len(time_code)):
            for j in range(len(time_code[i])):
                time_code[i][j] += -1

    y = data_obj.data_table[:, 0].todense()
    if target_in_training: 
        X = data_obj.data_table.to_scipy_sparse()
        X = X.tocsr()
        X[:, 0] = 2 #set to mask token 
        X = X.tocoo()

    else:
        X = data_obj.data_table[:, 1:].to_scipy_sparse()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    X_train = X_train.todense()
    X_test = X_test.todense()

    if return_tensors:
        return X_train, X_test, y_train, y_test, time_code

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))



    num_classes = 2
    input_size = 16278

    return input_size, num_classes, train_dataset, test_dataset, time_code














get_mimic(target_in_training=False)
print('ass')