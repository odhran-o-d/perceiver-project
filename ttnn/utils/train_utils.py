"""Utils for model/optimizer initialization and training."""
import numpy as np
from torch import optim
from transformers import AdamW
from ttnn.utils.optim_utils import Lookahead, RAdam, Ranger, Lamb
import pprint


def init_optimizer(c, model_parameters, device):
    # AdaFactor with HuggingFace default values
    # https://huggingface.co/transformers/main_classes/
    #   optimizer_schedules.html#adafactor-pytorch
    # TODO: Debug Adafactor optimizer
    # if c.exp_optimizer == 'adafactor':
    #     # Adafactor internally adjusts LR based on scale_parameter,
    #     # relative_step, and warmup_init options
    #       optimizer = Adafactor(
    #           params=model_parameters, weight_decay=c.exp_weight_decay)

    # Adam with weight decay fix from HuggingFace, as introduced in
    # Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
    if 'adam_weight_decay' in c.exp_optimizer:
        optimizer = AdamW(
            params=model_parameters, lr=c.exp_lr,
            weight_decay=c.exp_weight_decay)
    elif 'radam' in c.exp_optimizer:
        optimizer = RAdam(
            params=model_parameters, lr=c.exp_lr,
            weight_decay=c.exp_weight_decay, betas=(0.9, 0.999), eps=1e-8)
    elif c.exp_optimizer == 'ranger':  # Ranger includes lookahead
        optimizer = Ranger(
            params=model_parameters, k=c.exp_lookahead_update_cadence,
            N_sma_threshhold=5,
            betas=(.95, 0.999),  # They report .95 better than .9
            eps=1e-5, weight_decay=c.exp_weight_decay,
            use_gc=True  # Gradient centralization
        )
    # Adam parameters from Attention is All You Need
    # Don't recommend using with weight decay - WD will interact with the
    # Adam m and v parameters in strange ways as shown in above paper
    elif 'attention_is_all_you_need' in c.exp_optimizer:
        optimizer = optim.Adam(
            params=model_parameters, betas=[0.9, 0.98], eps=1e-9, lr=c.exp_lr,
            weight_decay=c.exp_weight_decay)
    elif 'default' in c.exp_optimizer:
        # TODO: consider using deepspeed library for GPU acceleration
        optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
    elif 'lamb' in c.exp_optimizer:
        # if device == 'cpu':
        #     lamb = Lamb
        # else:
            # from deepspeed.ops.lamb import FusedLamb
            # lamb = FusedLamb
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
            weight_decay=c.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError

    if c.exp_optimizer.startswith('lookahead_'):
        optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

    return optimizer


def multidim_intersect(arr1, arr2):
    """
    From https://stackoverflow.com/questions/
    9269681/intersection-of-2d-numpy-ndarrays
    """
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def get_sorted_params(model):
    param_count_and_name = []
    for n,p in model.named_parameters():
        if p.requires_grad:
            param_count_and_name.append((p.numel(), n))

    pprint.pprint(sorted(param_count_and_name, reverse=True))


def count_parameters(model):
    r"""
    Due to Federico Baldassarre
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
