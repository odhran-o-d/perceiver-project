from ttnn.datasets.base import BaseDataset
from pathlib import Path
import sparse
from os.path import join, isfile
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader, Subset


def get_cluster_dataset(seed=42):
    N_samples = 10
    K_train = 200
    K_test = 200
    input_dims = 32
    sigma = 0.1
    num_classes = 1
    """
    The cluster for the training and test set are disjunct.
    The validation set is the union of these two disjunct datasets.
    """

    g = torch.Generator()
    g.manual_seed(seed)
    D = input_dims
    N = N_samples

    # Generate the unique string indexing every cluster
    # repeat_interleave produces
    cluster_encodings_train = torch.randn(
        (K_train, D), generator=g).repeat_interleave(N, dim=0)
    cluster_encodings_test = torch.randn((K_test, D), generator=g)

    # Generate labels
    train_labels = torch.randn((K_train * N, 1), generator=g) * sigma
    train_offset = torch.linspace(-K_train / 2, K_train / 2 - 1,
                                    K_train).repeat_interleave(N)
    train_labels = train_labels.squeeze() + train_offset
    test_labels = torch.linspace(K_train / 2, K_train / 2 + K_test - 1, K_test)

    train_dataset = TensorDataset(
        cluster_encodings_train, train_labels.unsqueeze(1))
    test_dataset = TensorDataset(
        cluster_encodings_test, test_labels.unsqueeze(1))

    return input_dims, num_classes, train_dataset, test_dataset


def get_img_num_per_cls(img_max, cls_num, imb_type, imb_factor):
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):  # creates exp scale in datapoints
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':  # sets half of datapoints to imbalanced
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data(train_data, test_data, img_num_per_cls):
    # NOTE: this does not suffle - make sure input is shuffled
    train_labels = np.array(
        [train_data[i][1] for i in range(len(train_data))])
    test_labels = np.array(
        [test_data[i][1] for i in range(len(test_data))])
    scale_factor = len(train_labels) // len(test_labels)

    classes = np.unique(train_labels)
    train_indices = []
    test_indices = []

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        train_indices += np.where(
            train_labels == the_class)[0][:the_img_num].tolist()
        test_indices += np.where(
            test_labels == the_class)[0][:the_img_num//scale_factor].tolist()

    train_data = Subset(train_data, train_indices)
    test_data = Subset(test_data, test_indices)

    print('train data imbalance:')
    print(np.unique([
        train_data[x][1] for x in range(len(train_data))], return_counts=True))

    print('test data imbalance:')
    print(np.unique([
        test_data[x][1] for x in range(len(test_data))], return_counts=True))

    return train_data, test_data