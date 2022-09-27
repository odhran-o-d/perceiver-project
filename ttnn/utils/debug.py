import torch
import numpy as np


def modify_data(c, batch_dict, dataset_mode, num_steps):
    """Modify data for debugging row interactions in synthetic experiments."""

    if ((dis := c.debug_row_interactions_disable_after) != -1):
        if num_steps >= dis:
            return batch_dict

    if c.debug_row_interactions_mode == 'dice-roll':
        return dice_roll(c, batch_dict, dataset_mode)
    elif c.debug_row_interactions_mode == 'crossword':
        return crossword(c, batch_dict, dataset_mode)
    elif c.debug_row_interactions_mode == 'lookup':
        return lookup(c, batch_dict, dataset_mode)
    elif 'protein-duplicate' in c.debug_row_interactions_mode:
        return protein_duplicate(
            c, batch_dict, dataset_mode, c.debug_row_interactions_mode)
    else:
        raise ValueError


def corrupt_rows(c, batch_dict, dataset_mode, row_index):
    """Corrupt rows:
    (i) Duplication experiments -- find the duplicated row of the specified
        `row_index`. Flip its label.
    (ii) Standard datasets -- for each column, apply an independent permutation
        over entries in all rows other than row `row_index`.
    """
    if (c.debug_row_interactions and
            c.debug_row_interactions_mode == 'protein-duplicate'):
        return corrupt_duplicate_rows(c, batch_dict, dataset_mode, row_index)
    else:
        # Check this for multiple columns
        # (((modified_batch_dict['masked_tensors'][1]
        # - batch_dict_['masked_tensors'][1])**2).sum(-1) == 0).nonzero()
        # Check that this is only at row_index.
        # modified_batch_dict['train_mask_matrix'].nonzero()
        return corrupt_standard_dataset(c, batch_dict, dataset_mode, row_index)


def duplicate_batch_dict(batch_dict):
    def recursive_clone(obj):
        if isinstance(obj, (int, float)):
            return obj
        elif isinstance(obj, list):
            return [recursive_clone(elem) for elem in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.clone().detach()
        elif obj is None:
            return None
        else:
            raise NotImplementedError

    new_batch_dict = {}
    for key, value in batch_dict.items():
        new_batch_dict[key] = recursive_clone(value)

    return new_batch_dict


def corrupt_duplicate_rows(c, batch_dict, dataset_mode, row_index):
    """
    The aim of this corruption is to show that using the `designated lookup
    row` (located at `row_index`, which is a duplicate of the row at
    `row_index` + N) is necessary to solve the task for duplicated datasets,
    like protein-duplication.

    We wish to remove the ability to perform a successful lookup, and
    accomplish this by "flipping" the label of the duplicated row.
        - We can't simply input just a single row to our model, because
            we'd be dramatically changing batch statistics (and weird things
            may happen with our attention modules), destabilizing the model.
        - We don't want to corrupt the features as well -- the model should
            still be able to lookup the right row, but then should fail
            because of the label alteration we made.

    We will select a new label to which we flip the label of the designated
    lookup row by selecting uniformly at random from other unmasked rows.
    These unmasked rows are specified by the label_matrix, which is aware
    of stochastic label masking changes.

    Finally, we restrict the label_matrix to assure that we are only
    evaluating a loss on the `row_index`.
    """
    # Avoid overwriting things we will need in corruptions for other rows
    bd = duplicate_batch_dict(batch_dict)

    # if bd['augmentation_mask_matrix'] is not None:
    #     # TODO: can trivially extend this to feature augmentations by
    #     #  considering augmentation_mask_matrix also in the for-loop locations.
    #     raise NotImplementedError

    if bd['label_mask_matrix'] is not None:
        # Triggers for stochastic label masking.
        # Only not None in train mode. In which case we can use it to only
        # reveal those train indices that have been masked.
        label_matrix = 'label'
    else:
        # We are in val/test mode. In which case all val/test labels are masked
        # and need to be revealed at val/test time, to check that model is
        # actually learning interactions!
        label_matrix = dataset_mode

        # (Note that there may be stochastic masking on the train labels still.
        # but we do not reveal those anymore as there is no loss computed on
        # them.)

    if bd[f'{label_matrix}_mask_matrix'] is None:
        raise NotImplementedError

    num_cols = len(bd['data_arrs'])
    num_rows = bd['data_arrs'][0].shape[0] // 2

    # Keep track of target columns -- we will need to zero out the
    # label_matrix, and then set only the row_index in the specified
    # target columns so that we are only evaluating loss on our chosen
    # row index
    target_cols = []

    for col in range(num_cols):
        # get true values wherever the label matrix has masks
        locations = bd[f'{label_matrix}_mask_matrix'][:, col].nonzero(
            as_tuple=True)[0]
        if locations.nelement() == 0:
            continue

        target_cols.append(col)

        # These locations currently give us indexes where the loss should
        # be evaluated. We can determine the locations of the unmasked rows
        # by subtracting the original number of rows.
        locations -= num_rows

        # Next, we remove the provided row_index, as we do not want to flip its
        # label to itself -- this would of course be unsuccessful in corrupting
        # the label!
        locations = locations.tolist()
        locations = list(set(locations) - {row_index})

        # Randomly select one of the locations
        flip_index = np.random.choice(locations)

        # Replace the label of the `designated lookup row` with that of the
        # flip_index row we have just randomly selected
        bd[
            'masked_tensors'][col][row_index] = bd[
            'masked_tensors'][col][flip_index]

    # Only evaluate loss on the row_index in appropriate target columns.

    # Obtain loss index as originally specified row_index + number of rows
    loss_index = row_index + num_rows
    rows_to_zero = list(set(range(int(num_rows * 2))) - {loss_index})
    bd[f'{label_matrix}_mask_matrix'][rows_to_zero, :] = False

    # if dataset_mode != 'train':
    #     a = 1

    return bd


def corrupt_standard_dataset(c, batch_dict, dataset_mode, row_index):
    """
    The aim of this corruption is to show that using row interactions improves
    performance on a standard dataset, such as protein, higgs, or forest-cover.

    We cannot just input a single row to our model, because we'd be
    dramatically changing batch statistics (and weird things may happen with
    our attention modules), destabilizing the model.

    To accomplish this corruption, we independently permute each of the columns
    over all row indices, __excluding__ the specified row index.
    """
    # Avoid overwriting things we will need in corruptions for other rows
    bd = duplicate_batch_dict(batch_dict)

    n_cols = len(bd['data_arrs'])
    n_rows = bd['data_arrs'][0].shape[0]

    # Row indices to shuffle -- exclude the given row_index
    row_indices = list(set(range(n_rows)) - {row_index})

    # Shuffle all rows other than our selected one, row_index
    # Perform an independent permutation for each column so the row info
    # is destroyed (otherwise, our row-equivariant model won't have an
    # issue with permuted rows).
    for col in range(n_cols):
        # Test -- if we ablate shuffle, do not swap around elements
        if not c.debug_corrupt_standard_dataset_ablate_shuffle:
            shuffled_row_indices = np.random.permutation(row_indices)

            # Shuffle masked_tensors, which our model sees at input.
            # Don't need to shuffle data_arrs, because the row at which
            # we evaluate loss will be in the same place.
            bd['masked_tensors'][col][row_indices] = bd[
                'masked_tensors'][col][shuffled_row_indices]

        # We also zero out the
        # {dataset_mode}, augmentation, and label mask matrices at all
        # rows other than row_index
        for matrix in [dataset_mode, 'augmentation', 'label']:
            mask = f'{matrix}_mask_matrix'
            if bd[mask] is not None:
                bd[mask][:, col][row_indices] = False

    return bd


def dice_roll(c, batch_dict, dataset_mode):
    """Replace target column by copies from a single dice roll."""

    # first: create the true data for current batch
    col = batch_dict['data_arrs'][0]
    col[:] = 0
    # roll dice
    idx = np.random.choice(range(6))
    col[:, idx] = 1
    col[:, -1] = 0  # (there's currently a bug in data loader)

    # second: create the masked version of the input data
    mt = batch_dict['masked_tensors'][0]
    mt[:] = col
    # mask out the second half of the data (this is the prediction target)
    cut = mt.shape[0] // 2
    mt[cut:] = 0
    mt[cut:, -1] = 1

    # now the second half of the data is used as train/val/test set
    # want to predict these masked out values
    # (the matrices tell you where to compute loss in the specific dataset
    # modes. doesnt matter here as there is no overfitting)

    m = batch_dict[f'{dataset_mode}_mask_matrix'][:, 0]
    # compute loss on second half
    m[:cut] = False
    m[cut:] = True

    return batch_dict


def crossword(c, batch_dict, dataset_mode):
    """First generate a list of random data without any masking. 
    Then, create a copy of that data and mask out single columns.
    Only by looking into other rows and finding original data can this be
    solved.
    """
    # I think it's probably easiest to do this column wise

    constant_target_col = True

    # D constant for debug data
    N, D = batch_dict['data_arrs'][0].shape
    if dataset_mode != 'train':
        # a = 1
        # there's currently a bug in the dataloader where test/val time
        # only put in test/val. which sucks for me
        # disable everything for these...
        # don't look at val and test loss
        return batch_dict

    N_out = 12
    N = N_out
    num_cols = len(batch_dict['data_arrs'])

    # cut off data at n_out
    for col in range(num_cols):
        batch_dict['data_arrs'][col] = batch_dict['data_arrs'][col][:N_out]
        batch_dict['masked_tensors'][col] = (
            batch_dict['masked_tensors'][col][:N_out])

    batch_dict[f'{dataset_mode}_mask_matrix'] = (
        batch_dict[f'{dataset_mode}_mask_matrix'][:N_out])

    first_half = torch.arange(0, N//2)
    second_half = torch.arange(N//2, N)

    # for each row in second half, sample a column in which to corrupt
    if not constant_target_col:
        random_feats = torch.randint(
            low=0,
            high=num_cols,  # high is exclusive, but we still dont want to set mask
            size=[N//2],
            requires_grad=False)
    else:
        random_feats = torch.zeros(size=[N//2])

    for col in range(num_cols):

        bdc = batch_dict['data_arrs'][col]
        mt = batch_dict['masked_tensors'][col]

        # reset data
        bdc[:] = 0

        # for each row sample a random column to fill
        random_cols = torch.randint(
            low=0,
            high=D-1,  # high is exclusive, but we still dont want to set mask
            size=[N//2],
            requires_grad=False)
        bdc[torch.arange(0, N//2), random_cols] = 1

        # copy data into second half
        bdc[N//2:] = bdc[:N//2]

        # copy into masked tensors
        mt[:] = bdc

        # now mask out parts of that data
        # get row indices for which we want to mask this column out
        mask_out = second_half[random_feats == col]
        mt[mask_out] = 0
        mt[mask_out, -1] = 1

        # now set masks active where we mask out
        m = batch_dict[f'{dataset_mode}_mask_matrix'][:, col]
        m[:] = 0
        m[mask_out] = 1
        # torch.nonzero(m) == mask_out

    # now apply a random permutation to all rows, s.t. model cannot remember
    # matching between bottom and top half (although our model does not know
    # about row indices anyways, so this is slightly useless)

    return random_row_perm(N, batch_dict, dataset_mode)
    # return batch_dict


def lookup(c, batch_dict, dataset_mode):
    """A bit simpler than crossword, since not multiple values need to be
    compared. Instead it's enough to compare one column.

    """

    N_in, D = batch_dict['data_arrs'][0].shape
    num_cols = len(batch_dict['data_arrs'])

    if dataset_mode != 'train':
        # a = 1
        # there's currently a bug in the dataloader where test/val time
        # only put in test/val. which sucks for me
        # disable everything for these...
        # don't look at val and test loss
        return batch_dict

    N_out = 12
    # cut off data at n_out
    for col in range(num_cols):
        batch_dict['data_arrs'][col] = batch_dict['data_arrs'][col][:N_out]
        batch_dict['masked_tensors'][col] = (
            batch_dict['masked_tensors'][col][:N_out])

    batch_dict[f'{dataset_mode}_mask_matrix'] = (
        batch_dict[f'{dataset_mode}_mask_matrix'][:N_out])

    # A) write true data
    # A) 1) write index into first col
    bdc = batch_dict['data_arrs'][0]
    bdc[:] = 0
    bdc[:N_out//2] = torch.cat([
        torch.eye(N_out//2),
        torch.zeros(N_out//2, 1)], 1)  # zeros are for masks
    bdc[N_out//2:] = bdc[:N_out//2]

    # A) 2) some random matchings of indices into second col
    bdc = batch_dict['data_arrs'][1]
    bdc[:] = 0
    bdc[:N_out//2] = torch.cat([
        torch.eye(N_out//2)[torch.randperm(N_out//2)],
        torch.zeros(N_out//2, 1)], 1)
    bdc[N_out//2:] = bdc[:N_out//2]

    # Now copy this into masked data
    batch_dict['masked_tensors'][0][:] = batch_dict['data_arrs'][0].clone()
    batch_dict['masked_tensors'][1][:] = batch_dict['data_arrs'][1].clone()

    # mask out second half of second column
    batch_dict['masked_tensors'][1][N_out//2:] = 0
    batch_dict['masked_tensors'][1][N_out//2:, -1] = 1

    # write correct mask matrices
    # first zero them out
    batch_dict[f'{dataset_mode}_mask_matrix'][:] = 0
    # then set masks
    batch_dict[f'{dataset_mode}_mask_matrix'][N_out//2:, 1] = 1

    return random_row_perm(N_out, batch_dict, dataset_mode)
    # return batch_dict


def random_row_perm(N, batch_dict, dataset_mode):
    row_perm = torch.randperm(N)
    num_cols = len(batch_dict['data_arrs'])
    for col in range(num_cols):
        bdc = batch_dict['data_arrs'][col]
        bdc[:] = bdc[row_perm]

        mt = batch_dict['masked_tensors'][col]
        mt[:] = mt[row_perm]

    batch_dict[f'{dataset_mode}_mask_matrix'] = (
        batch_dict[f'{dataset_mode}_mask_matrix'][row_perm])

    return batch_dict


def leakage(c, batch_dict, masked_tensors, label_mask_matrix, dataset_mode):
    if c.data_set != 'breast-cancer':
        raise Exception

    if not (c.model_label_bert_mask_prob[dataset_mode] == 1):
        raise ValueError(
            'Leakage check only supported for deterministic label masking.')

    target_col = masked_tensors[0]
    assert target_col[:, -1].sum() == masked_tensors[0].size(0)
    assert target_col[:, 0].sum() == 0
    assert target_col[:, 1].sum() == 0
    assert label_mask_matrix is None

    n_label_loss_entries = batch_dict[
        f'{dataset_mode}_mask_matrix'].sum()

    print(f'{dataset_mode} mode:')
    print(f'Inputs over {masked_tensors[0].size(0)} rows.')
    print(
        f'Computing label loss at {n_label_loss_entries} entries.')


COL_LIM = 6


def protein_duplicate(c, batch_dict, dataset_mode, duplication_mode):
    """Append unmasked copy to the dataset.
    Allows for perfect loss if model exploits row interactions.
    This is version that respects dataset mode.
    Only unveil labels of current dataset mode.
    Currently does not unveil bert masks in copy.
    """

    N_in, D = batch_dict['data_arrs'][0].shape
    num_cols = len(batch_dict['data_arrs'])
    N_out = 2 * N_in
    bd = batch_dict

    if bd['label_mask_matrix'] is not None:
        # Triggers for stochastic label masking.
        # Only not None in train mode. In which case we can use it to only
        # reveal those train indices that have been masked.
        label_matrix = 'label'
    else:
        # We are in val/test mode. In which case all val/test labels are masked
        # and need to be revealed at val/test time, to check that model is
        # actually learning interactions!
        label_matrix = dataset_mode

        # (Note that there may be stochastic masking on the train labels still.
        # but we do not reveal those anymore as there is no loss computed on
        # them.)

    # do the same for each col
    for col in range(num_cols):

        # duplicate real data
        bd['data_arrs'][col] = torch.cat([
            bd['data_arrs'][col],
            bd['data_arrs'][col]], 0)

        # create new copy of data where masks are removed for everything that
        # is currently part of dataset_mode mask matrix
        # (i.e. all the labels)

        # append masked data again
        predict_rows = bd['masked_tensors'][col]
        if ('no-nn' in duplication_mode) and col > COL_LIM:
            lookup_rows = torch.zeros_like(predict_rows)
            lookup_rows[:, 0] = torch.normal(
                mean=torch.Tensor(N_in*[1.]),
                std=torch.Tensor(N_in*[1.]))

            bd['masked_tensors'][col] = torch.cat([
                lookup_rows, predict_rows], 0)
        else:
            lookup_rows = bd['masked_tensors'][col]
            bd['masked_tensors'][col] = torch.cat([
                lookup_rows, predict_rows], 0)

            # now unveil relevant values
            for matrix in [label_matrix, 'augmentation']:
                if bd[f'{matrix}_mask_matrix'] is None:
                    continue

                # get true values wherever current train/aug matrix has masks
                locations = bd[f'{matrix}_mask_matrix'][:, col].nonzero(
                    as_tuple=True)[0]
                # in these locations replace masked tensors with true data
                dtype = bd['masked_tensors'][col].dtype
                bd['masked_tensors'][col][locations] = (
                    bd['data_arrs'][col][locations].type(dtype))

        if ('target-add' in duplication_mode) and (col in bd['target_cols']):
            bd['masked_tensors'][col][locations] += 1


    # now modify the mask_matrices to fit dimensions of new data
    # (all zeros, don't need to predict on that new data)
    for matrix in [dataset_mode, 'augmentation', 'label']:
        mask = f'{matrix}_mask_matrix'
        if bd[mask] is not None:
            bd[mask] = torch.cat([
                torch.zeros_like(bd[mask]),
                bd[mask]], 0)

    return batch_dict


# old version: did not respect difference between unmasking train/test/val labels
# def protein_duplicate(c, batch_dict, dataset_mode):
#     """Append unmasked copy to the dataset.
#     Allows for perfect loss if model exploits row interactions."""

#     N_in, D = batch_dict['data_arrs'][0].shape
#     num_cols = len(batch_dict['data_arrs'])
#     N_out = 2 * N_in
#     bd = batch_dict

#     # do the same for each col
#     for col in range(num_cols):

#         # append real data unmasked
#         bd['masked_tensors'][col] = torch.cat([
#             bd['masked_tensors'][col],
#             bd['data_arrs'][col]], 0)

#         # btw checked that there a no weird cloning effects

#         # duplicate real data 
#         bd['data_arrs'][col] = torch.cat([
#             bd['data_arrs'][col],
#             bd['data_arrs'][col]], 0)

#     # now modify the mask_matrix
#     mask = f'{dataset_mode}_mask_matrix'
#     bd[mask] = torch.cat([
#         bd[mask],
#         torch.zeros_like(bd[mask])], 0)

#     # we're also doing augmentation masking in this one
#     mask = f'augmentation_mask_matrix'
#     if bd[mask] is not None:
#         bd[mask] = torch.cat([
#             bd[mask],
#             torch.zeros_like(bd[mask])], 0)

#     # we're not taking care of label_mask_matrix (only used for stochastic
#     # label masking)
#     if bd['label_mask_matrix'] is not None:
#         raise

#     return batch_dict
