import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Rectangle
import itertools
from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
import torch
from tqdm import tqdm

from ttnn.column_encoding_dataset import ColumnEncodingDataset, TTNNDataset
from ttnn.loss import Loss
from ttnn.optim import LRScheduler
from ttnn.optim import TradeoffAnnealer
from ttnn.utils import debug
from ttnn.utils.batch_utils import collate_with_pre_batching
from ttnn.utils.encode_utils import torch_cast_to_dtype
from ttnn.utils.eval_checkpoint_utils import EarlyStopCounter, EarlyStopSignal
from ttnn.utils.logging_utils import Logger
import seaborn as sns
import os


def plot_grid_query_pix(width, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_xticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.set_aspect(1)
    ax.set_yticks(np.arange(-width / 2, width / 2))  # , minor=True)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.grid(True, alpha=0.5)

    # query pixel
    querry_pix = Rectangle(xy=(-0.5,-0.5),
                          width=1,
                          height=1,
                          edgecolor="black",
                          fc='None',
                          lw=2)

    ax.add_patch(querry_pix);

    ax.set_xlim(-width / 2, width / 2)
    ax.set_ylim(-width / 2, width / 2)
    ax.set_aspect("equal")


def plot_attention_layer(attention_probs, axes):
    """Plot the 2D attention probabilities for a particular MAB attention map."""

    contours = np.array([0.9, 0.5])
    linestyles = [":", "-"]
    flat_colors = ["#3498db", "#f1c40f", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#34495e", "#1abc9c", "#95a5a6"]

    shape = attention_probs.shape
    num_heads, height, width = shape
    # attention_probs = attention_probs.reshape(width, height, num_heads)

    try:
        ax = axes[0]
    except:
        attention_prob_head = attention_probs[0].detach().cpu().numpy()
        sns.heatmap(attention_prob_head, ax=axes, square=True)
        axes.set_title(f'Head 1')
        return axes

    for head_index in range(num_heads):
        attention_prob_head = attention_probs[head_index].detach().cpu().numpy()
        sns.heatmap(attention_prob_head, ax=axes[head_index], square=True)
        axes[head_index].set_title(f'Head {head_index}')

    return axes

    # attention_at_center = attention_probs[width // 2, height // 2]
    # attention_at_center = attention_at_center.detach().cpu().numpy()
    # print(attention_at_center)
    #
    # # compute integral of distribution for thresholding
    # n = 1000
    # t = np.linspace(0, attention_at_center.max(), n)
    # integral = ((attention_at_center >= t[:, None, None, None]) * attention_at_center).sum(
    #     axis=(-1, -2)
    # )
    #
    # plot_grid_query_pix(width - 2, ax)
    #
    # for h, color in zip(range(num_heads), itertools.cycle(flat_colors)):
    #     f = interpolate.interp1d(integral[:, h], t, fill_value=(1, 0), bounds_error=False)
    #     t_contours = f(contours)
    #
    #     # remove duplicate contours if any
    #     keep_contour = np.concatenate([np.array([True]), np.diff(t_contours) > 0])
    #     t_contours = t_contours[keep_contour]
    #
    #     for t_contour, linestyle in zip(t_contours, linestyles):
    #         ax.contour(
    #             np.arange(-width // 2, width // 2) + 1,
    #             np.arange(-height // 2, height // 2) + 1,
    #             attention_at_center[h],
    #             [t_contour],
    #             extent=[- width // 2, width // 2 + 1, - height // 2, height // 2 + 1],
    #             colors=color,
    #             linestyles=linestyle
    #         )

# def plot_attention_positions_all_layers(
#         model, width, tensorboard_writer=None, global_step=None):
#
#     for layer_idx in range(len(model.encoder.layer)):
#         fig, ax = plt.subplots()
#         plot_attention_layer(model, layer_idx, width, ax=ax)
#
#         ax.set_title(f"Layer {layer_idx + 1}")
#         if tensorboard_writer:
#             tensorboard_writer.add_figure(f"attention/layer{layer_idx}", fig, global_step=global_step)
#         plt.close(fig)


def viz_att_maps(c, dataset, wandb_run, cv_index, n_splits):
    early_stop_counter = EarlyStopCounter(
        c=c, data_cache_prefix=dataset.model_cache_path,
        metadata=dataset.metadata,
        device=c.exp_device,
        wandb_run=wandb_run,
        cv_index=cv_index,
        n_splits=n_splits)

    # Initialize from checkpoint, if available
    num_steps = 0

    checkpoint = early_stop_counter.get_most_recent_checkpoint()
    if checkpoint is not None:
        checkpoint_epoch, (
            model, optimizer, _, num_steps) = checkpoint
    else:
        raise Exception('Could not find a checkpoint!')

    dataset.set_mode(mode='test', epoch=num_steps)
    batch_dataset = dataset.cv_dataset
    batch_dict = next(batch_dataset)

    from ttnn.utils import debug

    if c.debug_row_interactions:
        print('Detected debug mode.'
              'Modifying batch input to duplicate rows.')
        batch_dict = debug.modify_data(c, batch_dict, 'test', 0)

    # Run a forward pass
    masked_tensors = batch_dict['masked_tensors']
    masked_tensors = [
        masked_arr.to(device=c.exp_device)
        for masked_arr in masked_tensors]
    model.eval()
    model(masked_tensors)

    # Grab attention maps from SaveAttMaps modules
    # Collect metadata as we go
    layers = []

    # for current architecture, either 0 or 1.
    # e.g., in ISAB, the index 0 operation is H = MAB(ind, X),
    # and index 1 operation is X = MAB(X, H)
    mab_indices = []

    att_maps = []

    for name, param in model.named_parameters():
        if 'curr_att_maps' not in name:
            continue

        _, layer, mab_index, _, _ = name.split('.')
        layers.append(int(layer))

        try:
            mab_index_int = int(mab_index.split('mab')[1])
        except ValueError:
            mab_index_int = 0

        mab_indices.append(mab_index_int)
        att_maps.append(param)

    n_heads = c.model_num_heads

    from tensorboardX import SummaryWriter

    # create tensorboard writer
    # adapted from https://github.com/epfml/attention-cnn

    if not c.model_checkpoint_key:
        raise NotImplementedError

    save_path = os.path.join(c.viz_att_maps_save_path, c.model_checkpoint_key)
    tensorboard_writer = SummaryWriter(
        logdir=save_path, max_queue=100, flush_secs=10)
    print(f"Tensorboard logs saved in '{save_path}'")

    for i in range(len(att_maps)):
        layer_index = layers[i]
        mab_index = mab_indices[i]
        att_map = att_maps[i]

        # If n_heads != att_map.size(0), we have either nested attention,
        # or hybrid attention over the columns, which is applied to every
        # one of the batch dimension axes independently
        # e.g. we will have an attention map of shape (n_heads * N, D, D)
        # Just subsample a row for each head
        att_map_first_dim_size = att_map.size(0)
        if n_heads != att_map_first_dim_size:
            print('Subsampling attention over the columns.')
            print(f'Original size: {att_map.size()}')
            n_rows = att_map_first_dim_size // n_heads
            row_subsample_indices = []
            for row_index in range(0, att_map_first_dim_size, n_rows):
                row_subsample_indices.append(row_index)

            att_map = att_map[row_subsample_indices, :, :]
            print(f'Final size: {att_map.size()}')

        fig, axes = plt.subplots(ncols=n_heads, figsize=(15 * n_heads, 15))

        plot_attention_layer(
            att_map, axes=axes)
        if tensorboard_writer:
            tensorboard_writer.add_figure(
                f"attention/layer{layer_index}/mab_index{mab_index}", fig, global_step=1)
        plt.close(fig)
