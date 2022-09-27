# Convenience routines to analyse and restore TTNN Models
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from run import setup_args
from ttnn.column_encoding_dataset import ColumnEncodingDataset
from ttnn.utils.eval_checkpoint_utils import EarlyStopCounter
from ttnn.configs import build_parser


def t2n(t):
    return t.detach().cpu().numpy()


def get_c_and_wandb_from_cfg_string(string, offline=True):
    """Make sure string only has whitespaces between arguments."""
    parser = build_parser()
    args = parser.parse_args(args=string.replace('"', '').split(' '))
    args, wandb_args = setup_args(args)
    if offline:
        wandb_args.update(dict(mode="offline"))
    wandb_run = wandb.init(**wandb_args)
    args.cv_index = 0
    wandb.config.update(args, allow_val_change=True)
    c = wandb.config

    return c, wandb_run


def get_dataset(c, wandb_run, mode='test', cv_index=0):

    dataset = ColumnEncodingDataset(c)
    torch.manual_seed(c.torch_seed + cv_index)
    dataset.load_next_cv_split()
    # epoch argument only for logging
    dataset.set_mode(mode=mode, epoch=123)

    return dataset


def get_dataset_and_model(c, wandb_run, mode='test', cv_index=0):

    dataset = get_dataset(c, wandb_run, mode)

    early_stop_counter = EarlyStopCounter(
        c=c,
        data_cache_prefix=dataset.model_cache_path,
        metadata=dataset.metadata,
        device=c.exp_device,
        wandb_run=wandb_run,
        cv_index=cv_index,
        n_splits=min(dataset.n_cv_splits, c.exp_n_runs)
    )

    # Initialize from checkpoint, if available
    num_steps = 0

    checkpoint = early_stop_counter.get_most_recent_checkpoint()
    if checkpoint is not None:
        checkpoint_epoch, (
            model, optimizer, scaler, num_steps) = checkpoint
    else:
        raise Exception('Could not find a checkpoint!')

    if mode == 'train':
        model.train()
    else:
        print('Set model to eval mode.')
        model.eval()

    return dataset, model


def permutation_test_batch(model, batch_dict, c, verbose=False):
    """Permute input of model, see if output is permuted accordingly."""

    col = batch_dict['target_cols']

    if verbose:
        print(
            'Masked at',
            torch.nonzero(batch_dict['masked_tensors'][0][:, 1])[:, 0])
        print(
            'Predict at',
            torch.nonzero(batch_dict['test_mask_matrix'][:, 0])[:, 0])

    if c.debug_row_interactions:
        from ttnn.utils import debug
        if verbose:
            print(
                'Detected debug mode.'
                'Modifying batch input to duplicate rows.')
        batch_dict = debug.modify_data(c, batch_dict, 'test', 0)

    # entries without masks – these are entries that can be looked up
    revealed = torch.nonzero(batch_dict['masked_tensors'][0][:, 1] == 0)[:, 0]
    # prediction entries – these the model has to predict (mask == 1)
    pred_entries = torch.nonzero(batch_dict['test_mask_matrix'][:, 0])[:, 0]

    if verbose:
        print('Revealed Entries:', revealed)
        print('Predict at:', pred_entries)

    # Run a forward pass
    masked_tensors = batch_dict['masked_tensors']
    masked_tensors = [
        masked_arr.to(device=c.exp_device)
        for masked_arr in masked_tensors]
    out = model(masked_tensors)

    # Look at correlation between out and unmasked in (matching lookups).

    # Then permute inputs and see if output changes accordingly.

    N = out[0].shape[0]
    # check that revealed is correct
    assert np.array_equal(np.concatenate([
        np.arange(np.min(revealed.numpy())),
        revealed,
        np.arange(np.max(revealed.numpy()) + 1, N)], 0),
        np.arange(0, N))

    permutation = np.random.permutation(revealed)
    permuted_revealed = np.concatenate([
        np.arange(np.min(revealed.numpy())),
        permutation,
        np.arange(np.max(revealed.numpy()) + 1, N)], 0)

    # check that permutation revealed contains all values
    assert np.array_equal(np.sort(permuted_revealed), np.arange(N))

    in_permuted = [masked_tensors[0][permuted_revealed]] + masked_tensors[1:]
    out_permuted = model(in_permuted)

    if len(col) > 1:
        raise ValueError('Multiple target columns')

    out_dict = dict(
        in_data=t2n(masked_tensors[col[0]]),
        out=t2n(out[col[0]]),
        in_data_permuted=t2n(in_permuted[col[0]]),
        out_permuted=t2n(out_permuted[col[0]]),
        revealed=revealed,
        pred_entries=pred_entries,
        permutation=permutation)

    return out_dict


def replacement_test_batch(
        model, batch_dict, c, verbose=False, sweep_vals=None):
    """Replacem inputs of single target element, see if pred changes accord."""

    col = batch_dict['target_cols']
    if len(col) > 1:
        raise ValueError('Multi-col tests not implemented!')
    col = col[0]

    if col in batch_dict['cat_features']:
        data_type = 'classification'
    else:
        data_type = 'regression'
    print(f'Data type is >{data_type}<.')

    if verbose:
        print(
            'Masked at',
            torch.nonzero(batch_dict['masked_tensors'][col][:, -1])[:, 0])
        print(
            'Predict at',
            torch.nonzero(batch_dict['test_mask_matrix'][:, col])[:, 0])

    if c.debug_row_interactions:
        from ttnn.utils import debug
        if verbose:
            print(
                'Detected debug mode. '
                'Modifying batch input to duplicate rows.')
        batch_dict = debug.modify_data(c, batch_dict, 'test', 0)

    # entries without masks – these are entries that can be looked up
    revealed = torch.nonzero(
        batch_dict['masked_tensors'][col][:, -1] == 0)[:, 0]
    # prediction entries – these the model has to predict (mask == 1)
    pred_entries = torch.nonzero(batch_dict['test_mask_matrix'][:, col])[:, 0]

    # assert len(revealed) == len(pred_entries)
    # check that revealed data should actually be used for lookup
    # (i.e. actually corresponds to the copied data)
    # (this check may be a bit compute intensive to do always)
    # assert torch.all(
    #   torch.Tensor(
    #   [torch.equal(data_arr[revealed[idx]], data_arr[pred_entries[idx]])
    #    for data_arr in batch_dict['data_arrs']]))

    if verbose:
        print('Revealed Entries:', revealed)
        print('Predict at:', pred_entries)

    # get target values to sweep over
    if sweep_vals is None:
        min_val, max_val = [
            func(batch_dict['data_arrs'][col][:, 0])
            for func in [torch.max, torch.min]]
        new_values = torch.linspace(start=min_val, end=max_val, steps=10)
    else:
        new_values = sweep_vals

    # Run a forward pass
    masked_tensors = batch_dict['masked_tensors']
    masked_tensors = [
        masked_arr.to(device=c.exp_device)
        for masked_arr in masked_tensors]

    masked_tensors_orig = [t.clone() for t in masked_tensors]

    # just change the value of that single label it's looking up
    out_dicts = []
    for reveal, pred in zip(revealed, pred_entries):

        out_dict = dict(
            original_values=[],
            changed_lookup=[],
            changed_prediction=[])

        # clone s.t. we do not aggregate changes
        masked_tensors = [t.clone() for t in masked_tensors_orig]

        # record original lookup value
        out_dict['original_values'].append(
            get_original_value(masked_tensors[col][reveal], data_type).item())

        # cycle through replacement values
        for new_val in new_values:

            # change lookup value
            if data_type == 'regression':
                masked_tensors[col][reveal, 0] = new_val

            elif data_type == 'classification':
                masked_tensors[col][reveal, :-1] = 0
                masked_tensors[col][reveal, new_val] = 1

            else:
                raise ValueError

            out_dict['changed_lookup'].append(new_val.item())

            out = model(masked_tensors)
            changed_prediction = t2n(get_changed_value(
                out[col][pred], data_type))
            # record changed prediction
            out_dict['changed_prediction'].append(changed_prediction.item())

        out_dicts.append(out_dict)

    return out_dicts


def get_original_value(tensor, data_type):
    if data_type == 'regression':
        return tensor[0]
    elif data_type == 'classification':
        return torch.argmax(tensor[:-1])
    else:
        raise ValueError


def get_changed_value(tensor, data_type):
    if data_type == 'regression':
        return tensor
    elif data_type == 'classification':
        return torch.argmax(tensor)
    else:
        raise ValueError


def aggregate_over_epoch(
        batch_dataset, func, max_it=float('inf'), **kwargs):
    outs = []

    for batch_index, batch_dict in enumerate(batch_dataset):
        print(batch_index)
        kwargs.update(batch_dict=batch_dict)
        out = func(**kwargs)
        outs.append(out)

        if batch_index > max_it:
            print('Stopping execution')
            break

    return outs


def corrcoef(a, b):
    return np.corrcoef([a.reshape(-1), b.reshape(-1)])[0, 1]


def get_NN_list(bd, active_entry, return_mse=False,
                                   ignore_target_col=True, respect_train_test=False):
    """Sort entries based on feature space distance to active entry.

    Split into test and train entries.
    """
    target_col = bd['target_cols'][0]

    mse = []
    for it_col in range(len(bd['data_arrs'])):
        if ignore_target_col and it_col == target_col:
            continue

        data_col = bd['masked_tensors'][it_col][:, :-1]
        active = bd['masked_tensors'][it_col][active_entry, :-1]

        # sum over cat variables in a col
        mse += [((active - data_col)**2).sum(-1)]

    # sum over cols, --> get mse per row
    mse = t2n(torch.stack(mse, 1).sum(-1))

    nn_order = list(np.argsort(mse))

    if respect_train_test:
        train_order = [i for i in nn_order if i in test_entries]
        test_order = [i for i in nn_order if i not in test_entries]

        nn_order = np.concatenate([train_order, test_order], 0)

        assert np.array_equal(np.sort(nn_order), np.sort(nn_order))
        assert set(train_order).intersection(set(test_order)) == set()
        assert set(test_order).intersection(set(train_order)) == set()

    if return_mse:
        return nn_order, mse[nn_order]

    else:
        return nn_order


def get_NN_list_origin(bd, return_mse=False, ignore_target_col=True):
    """Sort entries based on feature space distance to active entry."""
    target_col = bd['target_cols'][0]

    mse = []
    for it_col in range(len(bd['data_arrs'])):
        if ignore_target_col and it_col == target_col:
            continue

        data_col = bd['masked_tensors'][it_col][:, :-1]

        # sum over cat variables in a col
        mse += [((data_col)**2).sum(-1)]

    # sum over cols, --> get mse per row
    mse = t2n(torch.stack(mse, 1).sum(-1))

    nn_order = list(np.argsort(mse))
    if return_mse:
        return nn_order, mse[nn_order]
    else:
        return nn_order


def permute_batch_dict(bd, new_order, dataset_mode):
    needs_permute = [
        'data_arrs', 'masked_tensors',
        f'{dataset_mode}_mask_matrix',
        'label_mask_matrix',
        'augmentation_mask_matrix']

    for data in needs_permute:
        if bd[data] is None:
            continue

        if not isinstance(bd[data], list):
            bd[data] = bd[data][new_order]

        else:
            for col in range(len(bd[data])):
                bd[data][col] = bd[data][col][new_order]


def dists_to_list(dists):
    N = dists.shape[0]
    pairwise_dists = [dists[i, j] for j in range(N) for i in range(j)]
    pairwise_dists = np.array(pairwise_dists)
    return pairwise_dists


def get_pairwise_mses(bd, ignore_target_col=True, col_lim=float('inf')):
    """Get pairwise mses between all input values."""
    target_col = bd['target_cols'][0]
    N = bd['data_arrs'][0].shape[0]

    mse = []
    for it_col in range(len(bd['masked_tensors'])):

        if ignore_target_col and it_col == target_col:
            continue

        if it_col > col_lim:
            break

        data_col = bd['masked_tensors'][it_col][:, :-1]

        diff = (
            data_col.unsqueeze(0).repeat(N, 1, 1)
            - data_col.unsqueeze(1).repeat(1, N, 1))

        # sum over cat variables in a col
        mse += [(diff**2).sum(-1)]

    # sum over cols, --> get mse per row
    mse = t2n(torch.stack(mse, -1).sum(-1))

    return mse


def sort_by_target_value(bd):

    masked_tensors = bd['data_arrs']
    target_col = bd['target_cols'][0]
    data = masked_tensors[target_col][:, 0]
    order = np.argsort(data)
    return order


def get_batch_apply_debug(c, wandb_run, batch_idx=0):

    dataset = get_dataset(c, wandb_run)
    batch_dataset = dataset.cv_dataset

    for i in range(batch_idx+1):
        batch_dict = next(batch_dataset)

    if c.debug_row_interactions:

        from ttnn.utils import debug
        print(
            'Detected debug mode.'
            'Modifying batch input to duplicate rows.')
        batch_dict = debug.modify_data(c, batch_dict, 'test', 0)

    return batch_dict


def get_data_from_mode(mode, dataset, sigma=False):
    Xy = []
    dataset.set_mode(mode=mode, epoch=0)
    bd = next(dataset.cv_dataset)

    for col in bd['num_features']:
        Xy.append(np.array(bd['data_arrs'][col][:, :-1]))

    Xy = np.concatenate(Xy, 1)
    if sigma:
        return Xy[:, 0], Xy[:, 1:], bd['sigmas'][bd['target_cols'][0]]

    return Xy[:, 0], Xy[:, 1:]


def get_mses(bd, active_entry, exp, print_targets=False, verbose=True):
    target_col = bd['target_cols'][0]
    D = len(bd['masked_tensors'])
    feature_cols = list(set(range(D)) - {target_col})
    n_rows = bd['test_mask_matrix'].shape[0]

    viable_subset = exp['viable_subset']
    deleted_rows = exp['deleted_rows']

    # make sure what we're seeing here is not just based on outlier statistics
    # draw a random subset from all rows the same size as viable subset
    random_subset = np.random.choice(
        list(set(range(n_rows)) - {active_entry}),
        len(viable_subset))

    # do extra for target column because target column is not visible to model
    unimp_diffs = []
    imp_diffs = []
    random_diffs = []

    for it_col in range(len(bd['data_arrs'])):

        data_col = bd['data_arrs'][it_col][:, :-1]

        active = data_col[active_entry]

        unimportant_rows = data_col[list(deleted_rows)]
        important_rows = data_col[list(viable_subset)]
        random_rows = data_col[list(random_subset)]

        unimp_diffs += [((active - unimportant_rows)**2).sum(-1)]
        imp_diffs += [((active - important_rows)**2).sum(-1)]
        random_diffs += [((active - random_rows)**2).sum(-1)]

    # mse of active against imp/uniomp per column
    unimp_diffs = torch.stack(unimp_diffs, 1)
    imp_diffs = torch.stack(imp_diffs, 1)
    random_diffs = torch.stack(random_diffs, 1)

    # now select columns, sum over them and take mean over rows
    # last statement: is difference(active, important) > difference
    # (active, unimportant)
    mean_imp = imp_diffs[:, feature_cols].sum(-1).mean()
    std_imp = imp_diffs[:, feature_cols].sum(-1).std()
    mean_unimp = unimp_diffs[:, feature_cols].sum(-1).mean()
    std_unimp = unimp_diffs[:, feature_cols].sum(-1).std()

    mean_random = random_diffs[:, feature_cols].sum(-1).mean()
    std_random = random_diffs[:, feature_cols].sum(-1).std()

    if verbose:
        print(f'MSE to kept features {mean_imp:.2f} +- {std_imp:.2f})')
        print(f'MSE to deleted features {mean_unimp:.2f} +- {std_unimp:.2f})')
        print(f'MSE to random features {mean_random:.2f} +- {std_random:.2f})')

    # are we choosing similar target values
    if print_targets:
        print(f'MSE to kept targets {imp_diffs[:, target_col].mean():.3f} +- '
              f'{imp_diffs[:, target_col].std():.3f})')
        print(f'MSE to deleted targets {unimp_diffs[:, target_col].mean():.3f} +- '
              f'{unimp_diffs[:, target_col].std():.3f})')
        print(f'MSE to random targets {random_diffs[:, target_col].mean():.3f} +- '
              f'{random_diffs[:, target_col].std():.3f})')

    return mean_imp, std_imp, mean_unimp, std_unimp, mean_random, std_random


def plot_mean_std(x, mean, std, c=None, dx=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if dx is None:
        dx = 0.1 * x

    ax.scatter(
        [x],
        [mean],
        color=c, zorder=10, **kwargs)

    ax.plot(
        [x - dx, x + dx],
        [mean, mean],
        c=c, **kwargs)

    ax.plot(
        2 * [x],
        [mean - std, mean + std],
        c=c, **kwargs)

    ax.plot(
        [x - dx, x + dx],
        [mean - std, mean - std],
        c=c, **kwargs)

    ax.plot(
        [x - dx, x + dx],
        [mean + std, mean + std],
        c=c, **kwargs)


def apply_duplicate_modification(x_test, y_test, duplicate_mode):
    from ttnn.utils.debug import COL_LIM
    """Code for DKL."""
    x_train = x_test
    y_train = y_test.clone()

    if 'target-add' in duplicate_mode:
        y_train = y_test + 1

    if 'no-nn' in duplicate_mode:
        x_train = x_train.clone()
        # protein has target column first. since this is now in y, we -1
        # x_train[:, COL_LIM-1:] = torch.normal(
        #     mean=1, std=1, size=x_train[:, COL_LIM-1:].shape)
        x_train[:, COL_LIM-1:] = torch.normal(
            mean=1, std=1, size=x_train[:, COL_LIM-1:].shape)

    # for normal protein duplication x_train = x_test
    return x_train, y_train
