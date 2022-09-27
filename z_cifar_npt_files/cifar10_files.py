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



def get_CIFAR10(root=".", data_imb=None, imb_type='exp'):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transforms_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ]

    train_transform = transforms.Compose(train_transforms_list)
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform,
        download=True
    )

    test_transforms_list = [
            transforms.ToTensor(),
            normalize,
    ]

    test_transform = transforms.Compose(test_transforms_list)
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform,
        download=True
    )

    if data_imb:
        imbalance = get_img_num_per_cls(
            img_max=len(train_dataset)//num_classes, cls_num=num_classes,
            imb_type=imb_type, imb_factor=data_imb)

        train_dataset, test_dataset = gen_imbalanced_data(
            train_dataset, test_dataset, imbalance)

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root=".", subpop_shift=None):
    input_size = 32
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transforms_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]
    train_transform = transforms.Compose(train_transforms_list)

    test_transforms_list = [
            transforms.ToTensor(),
            normalize]
    test_transform = transforms.Compose(test_transforms_list)

    if not subpop_shift:
        num_classes = 100
        train_dataset = datasets.CIFAR100(
            root + "data/CIFAR100", train=True, transform=train_transform,
            download=True)
        test_dataset = datasets.CIFAR100(
            root + "data/CIFAR100", train=False, transform=test_transform,
            download=True)
    else:
        n = subpop_shift
        num_classes = 20
        train_set = datasets.CIFAR100(
            root + "data/CIFAR100", train=True, transform=train_transform,
            download=True)
        test_set = datasets.CIFAR100(
            root + "data/CIFAR100", train=False, transform=test_transform,
            download=True)

        cf100_dic = {
            0: [72, 4, 95, 30, 55], 1: [73, 32, 67, 91, 1],
            2: [92, 70, 82, 54, 62], 3: [16, 61, 9, 10, 28],
            4: [51, 0, 53, 57, 83], 5: [40, 39, 22, 87, 86],
            6: [20, 25, 94, 84, 5], 7: [14, 24, 6, 7, 18],
            8: [43, 97, 42, 3, 88], 9: [37, 17, 76, 12, 68],
            10: [49, 33, 71, 23, 60], 11: [15, 21, 19, 31, 38],
            12: [75, 63, 66, 64, 34], 13: [77, 26, 45, 99, 79],
            14: [11, 2, 35, 46, 98], 15: [29, 93, 27, 78, 44],
            16: [65, 50, 74, 36, 80], 17: [56, 52, 47, 59, 96],
            18: [8, 58, 90, 13, 48], 19: [81, 69, 41, 89, 85]}

        coarse_map = {
            72: 0, 4: 0, 95: 0, 30: 0, 55: 0,
            73: 1, 32: 1, 67: 1, 91: 1, 1: 1,
            92: 2, 70: 2, 82: 2, 54: 2, 62: 2,
            16: 3, 61: 3, 9: 3, 10: 3, 28: 3,
            51: 4, 0: 4, 53: 4, 57: 4, 83: 4,
            40: 5, 39: 5, 22: 5, 87: 5, 86: 5,
            20: 6, 25: 6, 94: 6, 84: 6, 5: 6,
            14: 7, 24: 7, 6: 7, 7: 7, 18: 7,
            43: 8, 97: 8, 42: 8, 3: 8, 88: 8,
            37: 9, 17: 9, 76: 9, 12: 9, 68: 9,
            49: 10, 33: 10, 71: 10, 23: 10, 60: 10,
            15: 11, 21: 11, 19: 11, 31: 11, 38: 11,
            75: 12, 63: 12, 66: 12, 64: 12, 34: 12,
            77: 13, 26: 13, 45: 13, 99: 13, 79: 13,
            11: 14, 2: 14, 35: 14, 46: 14, 98: 14,
            29: 15, 93: 15, 27: 15, 78: 15, 44: 15,
            65: 16, 50: 16, 74: 16, 36: 16, 80: 16,
            56: 17, 52: 17, 47: 17, 59: 17, 96: 17,
            8: 18, 58: 18, 90: 18, 13: 18, 48: 18,
            81: 19, 69: 19, 41: 19, 89: 19, 85: 19,
        }

        train_labels = np.array(
            [train_set[i][1] for i in range(len(train_set))])
        test_labels = np.array(
            [test_set[i][1] for i in range(len(test_set))])

        classes = np.unique(train_labels)

        test_classes = sorted(
            np.concatenate([np.random.choice(
                cf100_dic[x], size=n, replace=False) for x in cf100_dic]))
        train_classes = list(set(classes).difference(set(test_classes)))

        train_in_train = np.concatenate([np.where(
            train_labels == x)[0] for x in train_classes]).tolist()
        train_in_test = np.concatenate([np.where(
            test_labels == x)[0] for x in train_classes]).tolist()
        test_in_train = np.concatenate([np.where(
            train_labels == x)[0] for x in test_classes]).tolist()
        test_in_test = np.concatenate([np.where(
            test_labels == x)[0] for x in test_classes]).tolist()

        for index, value in enumerate(test_set.targets):
            test_set.targets[index] = coarse_map[value]
        for index, value in enumerate(train_set.targets):
            train_set.targets[index] = coarse_map[value]

        test_dataset = torch.utils.data.ConcatDataset(
            [Subset(test_set, test_in_test), Subset(train_set, test_in_train)]
        )

        train_dataset = torch.utils.data.ConcatDataset(
            [Subset(test_set, train_in_test), Subset(train_set, train_in_train)]
        )

    return input_size, num_classes, train_dataset, test_dataset

