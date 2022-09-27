import numpy as np
import torch
import torch.distributions as D
from torch.distributions import MixtureSameFamily

from ttnn.datasets.base import BaseDataset


def sample(gmm, sample_shape=torch.Size()):
    with torch.no_grad():
        sample_len = len(sample_shape)
        batch_len = len(gmm.batch_shape)
        gather_dim = sample_len + batch_len
        es = gmm.event_shape

        # mixture samples [n, B]
        mix_sample = gmm.mixture_distribution.sample(sample_shape)
        mix_shape = mix_sample.shape

        # component samples [n, B, k, E]
        comp_samples = gmm.component_distribution.sample(sample_shape)

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim), mix_sample


class GMMDataset(BaseDataset):
    def __init__(self, c):
        N = 1000
        exp_test_perc = 0.2

        super().__init__(
            is_imputation=False,
            fixed_test_set_index=-int(N * exp_test_perc))

        self.N = N
        self.c = c

        # self.dirichlet_alpha = 1.5
        self.dirichlet_prior_alpha = None  # If None, uniform mixture weights

        self.base_dist_scale = 10
        self.n_comps = 5
        self.n_dims = 2
        self.comp_scale = 1  # Likelihood variance


    def load(self):
        if self.dirichlet_prior_alpha is None:
            # Uniform mixture weights
            self.mixture_weights = torch.ones(self.n_comps) / self.n_comps
        else:
            # Sample mixture weights from a Dirichlet prior
            dirichlet_prior = D.Dirichlet(
                concentration=torch.Tensor(
                    [self.dirichlet_prior_alpha]).repeat(self.n_comps))
            self.mixture_weights = dirichlet_prior.sample()

        mix = D.Categorical(self.mixture_weights)

        self.comp_locs = torch.normal(
            mean=0, std=self.base_dist_scale, size=(self.n_comps, self.n_dims))
        # self.comp_locs = torch.randn(self.n_comps, self.n_dims)

        if self.comp_scale is None:
            self.comp_scale_diag = torch.rand(self.n_comps, self.n_dims)
        else:
            self.comp_scale_diag = torch.Tensor(
                [self.comp_scale]).repeat(self.n_comps, self.n_dims)

        comp = D.Independent(D.Normal(
            self.comp_locs, self.comp_scale_diag), 1)

        gmm = MixtureSameFamily(mix, comp)

        train_perc = 1 - self.c.exp_val_perc - self.c.exp_test_perc
        assert 0 < train_perc < 1, 'Invalid train/val/test splits.'

        train_idx = int(self.N * train_perc)
        val_rows = int(self.N * self.c.exp_val_perc)
        test_rows = int(self.N * self.c.exp_test_perc)

        val_idx = train_idx + val_rows
        test_idx = val_idx + test_rows

        self.fixed_split_indices = [
            np.arange(0, train_idx),
            np.arange(train_idx, val_idx),
            np.arange(val_idx, test_idx)
        ]

        sample_shape = torch.Size([test_idx])
        self.N = test_idx

        data_table, comp_labels = sample(gmm, sample_shape)
        comp_labels = comp_labels.unsqueeze(-1)
        self.data_table = torch.cat(
            [data_table, comp_labels], axis=1).cpu().detach().numpy()

        self.D = self.n_dims + 1  # Add one for the label column
        self.cat_features = [self.D - 1]
        self.num_features = list(range(0, self.D - 1))

        # We only want to predict on the last column,
        # this is a categorical target
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        self.is_data_loaded = True

        self.tmp_file_or_dir_names = []
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    class Config():
        def __init__(self):
            self.exp_val_perc = 0.1
            self.exp_test_perc = 0.2

    c = Config()
    dataset = GMMDataset(c)
    dataset.load()

    dataset.mixture_weights
    data = dataset.data_table
    import pandas as pd

    df = pd.DataFrame(data)
    df.groupby(by=[2]).mean()
    dataset.comp_covar_diag