# from z_cifar_npt_files.fiddle_files import get_dense_mimic
import struct
import torch
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import TensorDataset
from data.mimic_subset.data_loaders import get_loaders_pheno
from z_cifar_npt_files.uci_data_loader import UCIDatasets
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split
from functools import partial
from torch.utils.data import DataLoader




def arrange_data(data, structure={'B':0,'N':1,'C':2}):
    assert len(structure) == len(data.shape)
    
    if len(data.shape) == 2:
        if 'C' in structure:
            return torch.unsqueeze(data, 1)
        else:
            return torch.unsqueeze(data, -1)
    if structure['C'] != data.shape[-1]:
        return torch.swapaxes(data, structure['C'], -1)


def load_mnist(batch_size):
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, test_dataset = random_split(dataset, [55000, 5000])

    loader = partial(
        DataLoader, batch_size=batch_size, num_workers=2, shuffle=True)


    train_loader, val_loader = list(map(loader, [train_dataset, test_dataset]))

    num_features = 1
    num_targets = 10
    time_code = None
    structure = {'B':0, 'C':1, 'H':2, 'W':3}

    return train_loader, val_loader, num_features, num_targets, time_code, structure



def load_mimic(batch_size, undersample=False):
    if undersample:
        X_train, X_test, y_train, y_test, time_code = get_dense_mimic(
            target_in_training=False, return_tensors=True)
        sampling = RandomUnderSampler()
        X_train, y_train = sampling.fit_resample(X_train, y_train)
        X_test, y_test = sampling.fit_resample(X_test, y_test)
        train_dataset = TensorDataset(
            torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = TensorDataset(
            torch.Tensor(X_test), torch.Tensor(y_test))

    else:
        _, _, train_dataset, test_dataset, time_code = get_dense_mimic(
            target_in_training=False)   

    loader = partial(
        DataLoader, batch_size=batch_size, num_workers=2, shuffle=True)

    train_loader, val_loader = list(map(loader, [train_dataset, test_dataset]))

    num_features = 2
    num_targets = 2
    structure = {'B':0, 'N':1}

    return train_loader, val_loader, num_features, num_targets, time_code, structure

    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    # val_loader = DataLoader(
    #     test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)


def load_uci(batch_size, dataset='uci-breast'):

    if dataset not in {
     'uci-housing', 'uci-concrete', 'uci-energy',
     'uci-power', 'uci-wine', 'uci-yacht', 'uci-breast'}:
        raise Exception('not known UCI dataset')
    data_obj = UCIDatasets(name=dataset[4:], n_splits=5)
    train_dataset = data_obj.get_split(train=True)
    test_dataset = data_obj.get_split(train=False)

    loader = partial(
        DataLoader, batch_size=batch_size, num_workers=2, shuffle=True)
    train_loader, val_loader = list(map(loader, [train_dataset, test_dataset]))

    num_features = next(iter(train_loader))[0].shape[-1]
    num_targets = 2 #todo check this
    time_code = None
    structure = {'B':0, 'C':1}

    return train_loader, val_loader, num_features, num_targets, time_code, structure


def load_mimic_pheno(batch_size, shuffle=False):
    train_loader, val_loader, test_loader = get_loaders_pheno(
        path='data/mimic_subset/pheno_data', 
        batch=batch_size, shuffle=shuffle)

    num_features = next(iter(train_loader))[0].shape[-1]
    num_targets = 25
    time_code = None
    structure = {'B':0, 'N':1, 'C':2}

    return train_loader, val_loader, num_features, num_targets, time_code, structure

   
