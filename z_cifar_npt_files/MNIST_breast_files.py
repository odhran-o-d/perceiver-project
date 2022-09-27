import argparse
import imp
import os
import pathlib
from ttnn.datasets.base import BaseDataset
from pathlib import Path
import sparse
from os.path import join, isfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotmap import DotMap
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import models, datasets, transforms
from tqdm import tqdm

from ttnn.model.ttnn_modules import SAB, SlotMAB, MAB
from ttnn.utils.train_utils import count_parameters
import math


def get_MNIST(root="."):
    input_size = 32
    num_classes = 10
    transform = transforms.Compose([
        # first, convert image to PyTorch tensor
        transforms.ToTensor(),
        # normalize inputs
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root + 'data/MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root + 'data/MNIST', train=False, transform=transform, download=True)

    return input_size, num_classes, train_dataset, test_dataset


def get_breast_cancer(test_split=0.20):
    input_size = 30
    num_classes = 2
    data_file_path = './data/breast-cancer/wisconsin_breast_cancer.csv'
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path):
        url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
               'breast-cancer-wisconsin/wdbc.data')

        df_cols = [
            "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean",
            "area_mean", "smoothness_mean", "compactness_mean",
            "concavity_mean", "concave points_mean", "symmetry_mean",
            "fractal_dimension_mean", "radius_se", "texture_se",
            "perimeter_se", "area_se", "smoothness_se", "compactness_se",
            "concavity_se", "concave points_se", "symmetry_se",
            "fractal_dimension_se", "radius_worst", "texture_worst",
            "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave points_worst",
            "symmetry_worst", "fractal_dimension_worst",
        ]

        print('Downloading data from %s...' % url)
        wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
        wis_df.to_csv(data_file_path)

    assert os.path.exists(data_file_path), \
        "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)
    wis_df['diagnosis'] = wis_df['diagnosis'].map({'M': 1, 'B': 0})

    X = wis_df.drop(['diagnosis'], axis=1).values
    y = wis_df['diagnosis'].values
    print(f"X.shape: {X.shape} | y.shape: {y.shape}")

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test))
    return input_size, num_classes, train_dataset, test_dataset
