import numpy as np
import torch

from ttnn.datasets.base import BaseDataset


class ClusterDataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            is_imputation=False,
            fixed_test_set_index=-c.cluster_K_test)

        self.c = c

        # Number of samples per cluster
        self.N = c.cluster_N_samples

        # Number of training set clusters
        self.K_train = c.cluster_K_train

        # Number of test set clusters;
        # these are disjunct from the training clusters
        self.K_test = c.cluster_K_test

        self.D = c.cluster_dimension  # Dimension of the encoding

        # Harcoded for now
        self.sigma = c.cluster_sigma

    def load(self, seed=42):
        """
        The cluster for the training and test set are disjunct.
        The validation set is the union of these two disjunct datasets.
        """

        print("Start generating cluster dataset")

        g = torch.Generator()
        g.manual_seed(seed)

        # Generate the unique string indexing every cluster,
        # and repeat each cluster string N times
        cluster_encodings_train = torch.randn(
            (self.K_train, self.D), generator=g).repeat_interleave(
                self.N, dim=0)
        cluster_encodings_test = torch.randn(
            (self.K_test, self.D), generator=g)
        cluster_encodings_test_in_val = (
            cluster_encodings_test.repeat_interleave(self.N, dim=0))
        cluster_encodings_val = torch.vstack(
            [cluster_encodings_train, cluster_encodings_test_in_val])

        # Generate labels
        train_labels = torch.randn((self.K_train * self.N, 1),
                                   generator=g) * self.sigma
        train_offset = torch.linspace(-self.K_train / 2, self.K_train / 2 - 1,
                                      self.K_train).repeat_interleave(self.N)
        train_labels = train_labels.squeeze() + train_offset

        val_labels = torch.randn(
            ((self.K_train + self.K_test) * self.N, 1),
            generator=g) * self.sigma
        val_offset = torch.linspace(
            -self.K_train / 2, self.K_train / 2 + self.K_test - 1,
            self.K_train + self.K_test).repeat_interleave(self.N)
        val_labels = val_labels.squeeze() + val_offset

        test_labels = torch.linspace(self.K_train / 2,
                                     self.K_train / 2 + self.K_test - 1,
                                     self.K_test)

        # Construct datasets
        train_data = torch.hstack(
            [cluster_encodings_train, train_labels.unsqueeze(1)])
        val_data = torch.hstack(
            [cluster_encodings_val, val_labels.unsqueeze(1)])
        test_data = torch.hstack(
            [cluster_encodings_test, test_labels.unsqueeze(1)])

        # Merge data
        self.data_table = torch.vstack(
            [train_data, val_data, test_data]).cpu().detach().numpy()
        n_train = self.K_train * self.N
        n_val = (self.K_train + self.K_test) * self.N
        n_test = self.K_test
        self.fixed_split_indices = [np.arange(0, n_train),
                                    np.arange(n_train, n_train + n_val),
                                    np.arange(n_train + n_val,
                                              n_train + n_val + n_test)]

        # Setup base information
        self.N = n_train + n_val + n_test
        self.D = self.D + 1 # Add one for the label collumn
        self.cat_features = []
        self.num_features = list(range(0, self.D))

        # We only want to predict on the last column,
        # this is a numerical target
        self.num_target_cols = [self.D - 1]
        self.cat_target_cols = []

        self.is_data_loaded = True

        self.tmp_file_or_dir_names = []
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    class Config():
        def __init__(self):
            self.cluster_N_samples = 10
            self.cluster_K_train = 200
            self.cluster_K_test = 20
            self.cluster_dimension = 32
            self.cluster_sigma = 0.1

    c = Config()
    dataset = ClusterDataset(c)
    dataset.load()

    data = dataset.data_table
    labels = data[:, -1]

    plt.figure(figsize=(100, 10))
    plt.hist(labels, bins=2000)
    plt.show()
