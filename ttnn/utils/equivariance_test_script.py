import torch
from ttnn.model.ttnn import TTNNModel
import numpy as np


def permuted_np_arr(dim) -> np.array:
    return np.random.permutation(dim).tolist()


def eq_test_script(c):
    print('Testing equivariance properties of model.')
    is_nested = c.model_use_nested_transformer

    # Set first, second, third dims for synthetic data
    # TODO: only works for default config args
    if is_nested:
        # Row, Col, Embed
        first_dim, second_dim, third_dim = 8, 10, 16
        input_feature_dims = [4, 4, 4, 4]
    else:
        # Batch, Row, Col * Embed
        first_dim, second_dim, third_dim = 8, 10, 64
        input_feature_dims = [4, 4, 4, 4]

    model = TTNNModel(c, input_feature_dims)

    # check for equivariance
    fake = torch.randn([first_dim, second_dim, third_dim]).float()

    # of first dim (if appropriate)
    first_dim_perm = permuted_np_arr(first_dim)
    print(
          'First dim equivariance: ',
          ((model.enc(fake[first_dim_perm]) -
           model.enc(fake)[first_dim_perm]) ** 2).sum())
    # --> approx 0 for flat, nested
    # (if permutations dont match, the squared sum is ~ 2000)

    # of second dim
    second_dim_perm = permuted_np_arr(second_dim)
    print(
        'Second dim equivariance: ',
        ((model.enc(fake[:, second_dim_perm]) -
          model.enc(fake)[:, second_dim_perm]) ** 2).sum())
    # --> approx 0 for flat, nested

    # of third dim
    third_dim_perm = permuted_np_arr(third_dim)
    print(
        'Third dim equivariance: ',
        ((model.enc(fake[:, :, third_dim_perm]) -
          model.enc(fake)[:, :, third_dim_perm]) ** 2).sum())
    # --> nonzero for both flat, nested

    # check no interactions
    print('First dim interactions (do not permute, but subset): ',
          ((model.enc(fake[:4]) -
            model.enc(fake)[:4]) ** 2).sum())
    # --> always zero for flat, nonzero for non-IMAB blocks + nested

    print('Second dim interactions (do not permute, but subset): ',
          ((model.enc(fake[:, :4])
            - model.enc(fake)[:, :4]) ** 2).sum())
    # --> nonzero for non-IMAB blocks + either flat, nested
