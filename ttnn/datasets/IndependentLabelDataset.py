import torch

import numpy as np

from ttnn.datasets.base import BaseDataset


class IndependentLabelDataset(BaseDataset):
    def __init__(self, c, N = 4096, D = 16):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=True)

        self.c = c
        self.N = N
        self.D = D

    def load(self, seed=42):
        """
        Generates the dataset based on the seed. By allowing a seed to be set
        we can generate multiple datasets to see if learning the lookup worked.
        """
        generator = torch.Generator()
        generator.manual_seed(seed)

        # The training dataset contains self.N*2 samples such that there is a duplicate for each training sample. For each pair of samples
        # the masking is made such that one of each pair is always unmasked. This allows for learning lookup.
        base_data = torch.randn((self.N, self.D), generator=generator)
        cloned = base_data.clone()

        # The validation and test dataset operate as eachothers duplicates. This way the validation dataset does not have a lookup
        # duplicate (becaude the test set is still masked at the time) while the test set can look at the validation set.
        val_data = torch.randn((self.N, self.D), generator=generator)
        test_data = val_data.clone()

        # Merge data
        self.data_table = torch.vstack([base_data, cloned, val_data, test_data])
        self.fixed_split_indices = [np.arange(0, self.N * 2), np.arange(self.N *2 , self.N * 3), np.arange(self.N * 3, self.N * 4)]

        # Setup base information
        self.N = self.N * 4
        self.cat_features = []
        self.num_features = list(range(0, self.D))

        # We only want to predict on the last collumn, this is a numerical target
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        self.is_data_loaded = True

        self.tmp_file_or_dir_names = []
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)


