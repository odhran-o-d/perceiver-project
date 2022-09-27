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
from einops import repeat



if torch.cuda.is_available():
    print(f'Running model with CUDA')
    exp_device = 'cuda:0'
else:
    print('Running model on CPU.')
    exp_device = 'cpu'



class TransformerBinaryEmbeddingBlock(torch.nn.Module):
    '''supports binary encoded data provided by
    the FIDDLE preprocessing system'''

    def __init__(
            self, dim_feature_embedding, time_code,
            train_label_masking_perc, label_embedding_dim=32,
            feature_embedding=True, time_embedding=True):
        super().__init__()

        self.feature_embedding = feature_embedding
        self.time_embedding = time_embedding
        self.time_code = time_code
        self.total_features = len(self.time_code[0]) + len(self.time_code[1])
        self.time_series_start = self.time_code[1][0]
        self.n_times = len(self.time_code) - 1

        self.feature_index_embedding = nn.Embedding(
            self.total_features, dim_feature_embedding)
        self.time_index_embedding = nn.Embedding(
            len(self.time_code), dim_feature_embedding)

        self.in_embedding = nn.ModuleList([
            nn.Linear(1, dim_feature_embedding)
            for dim_feature_encoding in range(self.total_features+1)])

        if train_label_masking_perc is None:
            label_embedding_dim = 0

        self.out_embedding = nn.ModuleList([
            nn.Linear(dim_feature_embedding+label_embedding_dim, 2)
            for dim_feature_encoding in range(self.total_features+1)])

    def forward(self, x):
        X = [self.in_embedding[t](torch.unsqueeze(x[:, t], 1))
             for t in self.time_code[0]]
        for time_subset in self.time_code[1:]:
            X += [self.in_embedding[self.time_code[1][i]](torch.unsqueeze(x[:, t], 1))
                  for i, t in enumerate(time_subset)]
        X = torch.stack(X, 1)

        # Compute feature index embeddings, and add them
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                torch.arange(self.total_features, device=exp_device))

            feature_index_embeddings = torch.cat((
                feature_index_embeddings[0:self.time_series_start],
                feature_index_embeddings[self.time_series_start:].repeat(
                    self.n_times, 1)))

            # Add a batch dimension (the rows)
            feature_index_embeddings = torch.unsqueeze(
                feature_index_embeddings, 0)

            # Tile over the rows
            feature_index_embeddings = feature_index_embeddings.repeat(
                X.size(0), 1, 1)

            # Add to X
            X = X + feature_index_embeddings

        if self.time_index_embedding is not None:
            time_index_embeddings = self.time_index_embedding(
                torch.arange(len(self.time_code), device=exp_device))

            time_index_embeddings = torch.cat(([
                repeat(time_index_embeddings[i], 'd -> n d', n=len(t))
                for i, t in enumerate(self.time_code)]))

            time_index_embeddings = torch.unsqueeze(
                time_index_embeddings, 0)

            time_index_embeddings = time_index_embeddings.repeat(
                X.size(0), 1, 1)

            X = X + time_index_embeddings

        return X

    def decode(self, x):
        X_ragged = [self.out_embedding[t](
            x[:, t]) for t in self.time_code[0]]
        for time_subset in self.time_code[1:]:
            X_ragged += [self.out_embedding[self.time_code[1][i]](
                x[:, t]) for i, t in enumerate(time_subset)]
        X = torch.stack(X_ragged, 1)
        X = F.log_softmax(X, dim=-1)
        # X = torch.argmax(X, dim=-1)
        return X


class ResNet18Encoder(torch.nn.Module):
    """From
    https://github.com/y0ast/pytorch-snippets/blob/
    main/minimal_cifar/train_cifar.py,
    Due to Joost van Amersfoort (OATML Group)
    Minimal script to train ResNet18 to 94% accuracy on CIFAR-10.
    """
    def __init__(self, in_channels=3, encoding_dims=128, pretrained=False,
                 apply_log_softmax=True):
        super().__init__()
        if pretrained:
            # Need to give it the ImageNet dimensions
            self.resnet = models.resnet18(
                pretrained=pretrained, num_classes=1000)
        else:
            self.resnet = models.resnet18(
                pretrained=pretrained, num_classes=encoding_dims)

        self.resnet.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()
        self.apply_log_softmax = apply_log_softmax

        # Replace last layer
        if pretrained:
            self.resnet.fc = nn.Linear(512, encoding_dims)

    def forward(self, x):
        x = self.resnet(x)
        if self.apply_log_softmax:
            x = F.log_softmax(x, dim=1)

        return x


"""NPT Variants. Perform ABD."""


class NPTLite(torch.nn.Module):
    def __init__(
            self, sab_args, stacking_depth=2, dim_hidden=128, dim_output=10):
        super().__init__()
        enc = []
        for _ in range(stacking_depth):
            enc.append(
                SAB(dim_hidden, dim_hidden, dim_hidden, sab_args))
        enc.append(nn.Linear(dim_hidden, dim_output))
        self.enc = nn.Sequential(*enc)

    def forward(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class InducingNPT(torch.nn.Module):
    """
    First apply cross-attention
    MAB(I, X)
    to obtain an intermediate representation of dimensions
    M x (D * E)

    Then repeatedly apply self-attention in this smaller
    space, eventually obtaining final inducing representation Z.

    Finally, apply cross-attention to project to output dims.
    MAB(X, Z)
    """
    def __init__(
            self, args, stacking_depth=2, dim_hidden=128, dim_output=10):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        num_inds = args['model_num_inds']

        # MAB(I, X)
        enc = [SlotMAB(dim_hidden, dim_hidden, dim_hidden,
                       args, num_inds=num_inds)]

        # Repeated SABs
        for i in range(1, stacking_depth - 1):
            enc.append(SAB(
                dim_hidden, dim_hidden, dim_hidden, args))

        self.enc = nn.Sequential(*enc)

        # MAB(X, Z)
        self.final_mab = MAB(
            dim_hidden, dim_hidden, dim_hidden, dim_hidden, args)

        self.final_linear = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):  # TODO: check against your perceiver and confirm
        x_new = self.enc(x)
        x = self.final_mab(x, x_new)
        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args['hidden_size']
        self.dropout = nn.Dropout(0)
        self.mask_identity = args['mask_identity']
        self.device = args['exp_device']

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = torch.matmul(q / self.hidden_size, k.transpose(-2, -1))

        if self.mask_identity:
            mask_value = -torch.finfo(attn.dtype).max
            batch_size, seq_length, _ = attn.size()
            mask = torch.eye(seq_length, seq_length).to(torch.bool)
            mask = torch.unsqueeze(mask, dim=0).to(self.device)
            attn = attn.masked_fill(mask, mask_value)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class BertSelfAttention(nn.Module):
    """From https://github.com/dat821168/multi-head_self-attention/blob/master/selfattention.py"""
    def __init__(self, args):
        super().__init__()
        assert args["hidden_size"] % args[
            "model_num_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = args['model_num_heads']
        self.attention_head_size = int(args['hidden_size'] / args['model_num_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args['hidden_size'], self.all_head_size)
        self.key = nn.Linear(args['hidden_size'], self.all_head_size)
        self.value = nn.Linear(args['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(args['hidden_size'], args['hidden_size'])

        self.mask_identity = args['mask_identity']
        self.device = args['exp_device']

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        if self.mask_identity:
            mask_value = -torch.finfo(attention_scores.dtype).max
            batch_size, num_heads, seq_length, _ = attention_scores.size()
            mask = torch.tile(
                torch.eye(seq_length, seq_length).to(torch.bool),
                (batch_size, num_heads, 1, 1)).to(self.device)
            attention_scores = attention_scores.masked_fill(mask, mask_value)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return output


class SABEnforcedNPT(torch.nn.Module):
    def __init__(
            self, args, stacking_depth=2, dim_hidden=128, dim_output=2):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        self.self_att = BertSelfAttention(
            args={'model_num_heads': args['model_num_heads'],
                  'hidden_size': dim_hidden,
                  'mask_identity': True,
                  'exp_device': args['exp_device']})
        self.final_linear = nn.Linear(dim_hidden, dim_output)

        heads = args['model_num_heads']
        print(f'Initialized SAB ({heads} heads) with masked identity.')

    def forward(self, x):
        x = self.self_att(x)
        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class SABEnforcedNPTSimple(torch.nn.Module):
    def __init__(
            self, args, stacking_depth=2, dim_hidden=128, dim_output=2):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        assert args['model_num_heads'] == 1

        self.self_att = ScaledDotProductAttention(
            args={'hidden_size': dim_hidden,
                  'mask_identity': True,
                  'exp_device': args['exp_device']})
        self.final_linear = nn.Linear(dim_hidden, dim_output)

        print('Initialized SAB-Simple (1 head) with masked identity.')

    def forward(self, x):
        x = self.self_att(x)
        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class SABNPT(torch.nn.Module):
    def __init__(
            self, args, stacking_depth=2, dim_hidden=128, dim_output=2):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        self.self_att = BertSelfAttention(
            args={'model_num_heads': args['model_num_heads'],
                  'hidden_size': dim_hidden,
                  'mask_identity': False,
                  'exp_device': args['exp_device']})
        self.final_linear = nn.Linear(dim_hidden, dim_output)

        heads = args['model_num_heads']
        print(f'Initialized SAB ({heads} heads) with un-masked identity.')

    def forward(self, x):
        x = self.self_att(x)
        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class SABNPTSimple(torch.nn.Module):
    def __init__(
            self, args, stacking_depth=2, dim_hidden=128, dim_output=2):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        assert args['model_num_heads'] == 1

        self.self_att = ScaledDotProductAttention(
            args={'hidden_size': dim_hidden,
                  'mask_identity': False,
                  'exp_device': args['exp_device']})
        self.final_linear = nn.Linear(dim_hidden, dim_output)

        print('Initialized SAB-Simple (1 head) with un-masked identity.')

    def forward(self, x):
        x = self.self_att(x)
        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class StrictNPT(torch.nn.Module):
    """
    Experimental:

    Strictly enforce nonparametric prediction.
    Needs two ResNets, that can optionally be tied.

    The key is to use a modified final layer that requires
    each point to be predicted as some nonlinear function of a
    convex combination of inducing points, WITH NO RESIDUAL CONNECTION.
    """
    def __init__(
            self, args, resnet_args,
            stacking_depth=2, dim_hidden=128, dim_output=10):
        super().__init__()

        if stacking_depth == 1:
            raise NotImplementedError

        num_inds = args['model_num_inds']

        # MAB(I, X)
        enc = [SlotMAB(dim_hidden, dim_hidden, dim_hidden,
                       args, num_inds=num_inds)]

        # Repeated SABs
        for i in range(1, stacking_depth - 1):
            enc.append(SAB(
                dim_hidden, dim_hidden, dim_hidden, args))

        self.enc = nn.Sequential(*enc)

        # MAB(X, Z)
        self.no_res_mab = MAB(
            dim_hidden, dim_hidden, dim_hidden, dim_hidden, args,
            ablate_res=True)  # Ablate the residual connection.

        self.final_linear = nn.Linear(dim_hidden, dim_output)

        self.last_layer_resnet = ResNet18Encoder(**resnet_args)

    def forward(self, x_npt, x):
        # Extracts inducing points.
        # TODO(nband): add recurrent encoder to extract better inducing points.
        # TODO(nband): consider adding a ResNet embedding for the
        #   inducing points -- could this result in interpretable inputs?
        # TODO(nband): implement lightweight prediction using inducing points.
        # z: inducing points
        z = self.enc(x_npt)

        # x_q: used to query the inducing points in final layer
        x_q = self.last_layer_resnet(x)
        x_q = torch.unsqueeze(x_q, 0)

        # Strictly non-parametric layer
        x = self.no_res_mab(x_q, z)

        x = self.final_linear(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)