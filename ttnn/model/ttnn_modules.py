"""Contains base attention modules."""

import math
import os

import torch
import torch.nn as nn

from ttnn.utils.param_init_utils import xavier_uniform_loc_


class SaveAttMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.curr_att_maps = None
        self.Q = None
        self.K = None
        self.V = None
        self.out = None
        self.out_pre_res = None

    def forward(self, X, Q, K, V):
        self.curr_att_maps = nn.Parameter(X)
        self.Q = nn.Parameter(Q)
        self.K = nn.Parameter(K)
        self.V = nn.Parameter(V)

        return X


class DenseGatedGeluDense(nn.Module):
    """
    Due to Huggingface's implementation of Google's T5.
    https://github.com/huggingface/transformers/blob/948b730f9777174335812cf7
    6de2a9dd9e4cf20e/src/transformers/models/t5/modeling_t5.py
    See also Shazeer 2020 (https://arxiv.org/pdf/2002.05202.pdf).
    Fixed to a 4x expansion factor, and depth 1.
    """

    def __init__(self, dim_out, dropout):
        super().__init__()
        self.wi_0 = nn.Linear(dim_out, dim_out * 4, bias=False)
        self.wi_1 = nn.Linear(dim_out, dim_out * 4, bias=False)
        self.wo = nn.Linear(dim_out * 4, dim_out, bias=False)
        self.dropout = dropout

    def gelu_new(self, x):
        """
        Implementation of the GELU activation function currently in Google
        BERT repo (identical to OpenAI GPT). Also see the Gaussian Error
        Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        return 0.5 * x * (
                1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_new(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class MAB(nn.Module):
    """Multi-head Attention Block."""
    def __init__(
            self, dim_Q, dim_KV, dim_emb, dim_out, c, dim_att=None,
            force_ablate_rff=False, ablate_res=False):
        """

        Inputs have shape (B_A, N_A, F_A), where
        * `B_A` is a batch dimension, along we parallelise computation,
        * `N_A` is the number of samples in each batch, along which we perform
        attention, and
        * `F_A` is dimension of the embedding at input
            * `F_A` is `dim_Q` for query matrix
            * `F_A` is `dim_KV` for key and value matrix.

        Q, K, and V then all get embedded to `dim_emb`.
        `dim_out` is the output dimensionality of the MAB which has shape
        (B_A, N_A, dim_out), this can be different from `dim_KV` due to
        the head_mixing.

        This naming scheme is inherited from set-transformer paper.

        dim_att: Tuple[int, int], needs to be specified when we aim to learn
            and apply either additive encodings to the attention weight tensor
            (pre-softmax) or multiplicative encodings to the attention score
            tensor (post-softmax).
            NOTE: this is only valid when performing attention over the
            columns, as in nested attention (else it would break row
            equivariance).
        force_ablate_rff: bool, if True, do not apply the rFF on this MAB.
        ablate_res: bool, if True, ablate residual connections.
        """
        super(MAB, self).__init__()
        mix_heads = c.model_mix_heads
        num_heads = c.model_num_heads
        sep_res_embed = c.model_sep_res_embed
        ln = c.model_att_block_layer_norm
        rff_depth = c.model_rff_depth
        self.att_score_norm = c.model_att_score_norm
        self.pre_layer_norm = c.model_pre_layer_norm
        self.rff_gated_gelu = c.model_rff_gated_gelu
        self.viz_att_maps = c.viz_att_maps
        self.model_ablate_rff = c.model_ablate_rff
        self.force_ablate_rff = force_ablate_rff
        self.ablate_res = ablate_res
        self.model_share_qk_sab_embedding = c.model_share_qk_sab_embedding

        if self.viz_att_maps:
            self.save_att_maps = SaveAttMaps()

        if dim_out is None:
            dim_out = dim_emb
        elif (dim_out is not None) and (mix_heads is None):
            print('Warning: dim_out transformation does not apply.')
            dim_out = dim_emb

        self.num_heads = num_heads
        self.dim_KV = dim_KV
        self.dim_split = dim_emb // num_heads

        if self.model_share_qk_sab_embedding:
            self.fc_qk = nn.Linear(dim_Q, dim_emb)
        else:
            self.fc_q = nn.Linear(dim_Q, dim_emb)
            self.fc_k = nn.Linear(dim_KV, dim_emb)

        self.fc_v = nn.Linear(dim_KV, dim_emb)

        self.fc_mix_heads = nn.Linear(dim_emb, dim_out) if mix_heads else None
        self.fc_res = nn.Linear(dim_Q, dim_out) if sep_res_embed else None

        # Initialize additive and multiplicative encodings
        self.init_additive_multiplicative_encodings(c, dim_att)

        if ln:
            if self.pre_layer_norm:  # Applied to X
                self.ln0 = nn.LayerNorm(dim_Q, eps=c.model_layer_norm_eps)
            else:  # Applied after MHA and residual
                self.ln0 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)

            self.ln1 = nn.LayerNorm(dim_out, eps=c.model_layer_norm_eps)
        else:
            self.ln0 = None
            self.ln1 = None

        self.hidden_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        self.att_scores_dropout = (
            nn.Dropout(p=c.model_att_score_dropout_prob)
            if c.model_att_score_dropout_prob else None)

        if not self.model_ablate_rff and not self.force_ablate_rff:
            if self.rff_gated_gelu:
                self.rff = DenseGatedGeluDense(
                    dim_out=dim_out, dropout=self.hidden_dropout)
            else:
                self.init_rff(dim_out, rff_depth)

    def init_additive_multiplicative_encodings(self, c, dim_att):
        att_additive_encoding = None
        att_multiplicative_encoding = None

        if dim_att is not None:
            # dimension of attention
            if isinstance(dim_att, int):
                dims = (self.num_heads, dim_att, dim_att)
            else:
                dims = (self.num_heads, *dim_att)

            if c.model_att_additive_encoding:
                att_additive_encoding = nn.Parameter(torch.Tensor(*dims))
                # Centered at 0
                nn.init.xavier_uniform_(att_additive_encoding, gain=1)

            if c.model_att_multiplicative_encoding:
                att_multiplicative_encoding = nn.Parameter(torch.Tensor(*dims))
                # Centered at 1 (defaults to identity)
                xavier_uniform_loc_(
                    att_multiplicative_encoding, loc=1, gain=1)

        self.att_additive_encoding = att_additive_encoding
        self.att_multiplicative_encoding = att_multiplicative_encoding

    def init_rff(self, dim_out, rff_depth):
        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        self.rff = [nn.Linear(dim_out, 4 * dim_out), nn.GELU()]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [nn.Linear(4 * dim_out, dim_out)]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        self.rff = nn.Sequential(*self.rff)

    def forward(self, X, Y):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X)
        else:
            X_multihead = X

        if self.model_share_qk_sab_embedding:
            Q = self.fc_qk(X_multihead)
        else:
            Q = self.fc_q(X_multihead)

        if not self.ablate_res:  # if self ablate res == false
            if self.fc_res is None:
                X_res = Q
            else:
                X_res = self.fc_res(X)  # Separate embedding for residual

        if self.model_share_qk_sab_embedding:
            K = self.fc_qk(Y)
        else:
            K = self.fc_k(Y)

        V = self.fc_v(Y)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        # TODO: track issue at
        # https://github.com/juho-lee/set_transformer/issues/8
        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = torch.einsum('ijl,ikl->ijk', Q_, K_)

        # Perform elementwise addition using learned "additive encodings" on
        # the pre-softmax attention weights.
        # These allow the model to, for example, focus on modelling some
        # interactions between columns while avoiding others entirely at
        # a particular row. Inspired by tree-based methods.
        if self.att_additive_encoding is not None:
            additive_stack = self.att_additive_encoding.repeat(
                int(A.size(0) / self.att_additive_encoding.size(0)), 1, 1)
            A = additive_stack + A

        if self.att_score_norm == 'softmax':
            A = torch.softmax(A / math.sqrt(self.dim_KV), 2)
        elif self.att_score_norm == 'constant':
            A = A / self.dim_split
        else:
            raise NotImplementedError

        # Perform elementwise multiplication using learned "multiplicative
        # encodings" on the post-softmax attention scores.
        # See above for explanation.
        if self.att_multiplicative_encoding is not None:
            mult_stack = self.att_multiplicative_encoding.repeat(
                int(A.size(0) / self.att_multiplicative_encoding.size(0)),
                1, 1)
            A = mult_stack * A

        if self.viz_att_maps:
            A = self.save_att_maps(A, Q_, K_, V_)

        # Attention scores dropout is applied to the N x N_v matrix of
        # attention scores.
        # Hence, it drops out entire rows/cols to attend to.
        # This follows Vaswani et al. 2017 (original Transformer paper).

        if self.att_scores_dropout is not None:
            A = self.att_scores_dropout(A)

        multihead = A.bmm(V_)
        multihead = torch.cat(multihead.split(Q.size(0), 0), 2)

        # Add mixing of heads in hidden dim.
        # TODO: ablate effect of this

        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead)
        else:
            H = multihead

        # Follow Vaswani et al. 2017 in applying dropout prior to
        # residual and LayerNorm
        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H)

        # True to the paper would be to replace
        # self.fc_mix_heads = nn.Linear(dim_V, dim_Q)
        # and Q_out = X
        # Then, the output dim is equal to input dim, just like it's written
        # in the paper. We should definitely check if that boosts performance.
        # This will require changes to downstream structure (since downstream
        # blocks expect input_dim=dim_V and not dim_Q)

        if not self.ablate_res: # if self ablate_res is false
            # Residual connection
            Q_out = X_res
            H = H + Q_out

        # Post Layer-Norm, as in SetTransformer and BERT.
        if not self.pre_layer_norm and self.ln0 is not None:
            H = self.ln0(H)

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H)
        else:
            H_rff = H

        if self.model_ablate_rff or self.force_ablate_rff:
            expanded_linear_H = H_rff
        else:
            # Apply row-wise feed forward network
            expanded_linear_H = self.rff(H_rff)

        if not self.ablate_res:   #if self ablate res equals false
            # Residual connection
            expanded_linear_H = H + expanded_linear_H

        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(expanded_linear_H)

        if self.viz_att_maps:
            self.save_att_maps.out = nn.Parameter(expanded_linear_H)
            self.save_att_maps.out_pre_res = nn.Parameter(H)

        return expanded_linear_H


class SAB(nn.Module):
    """Multi-head Self-Attention Block."""
    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, c, num_input_features=None):
        super(SAB, self).__init__()
        self.mab = MAB(
            dim_in, dim_in, dim_emb, dim_out, c, dim_att=num_input_features)

    def forward(self, X):
        return self.mab(X, X)






class IndepSAB(nn.Module):
    has_inducing_points = False

    def __init__(
            self, dim_in, dim_emb, dim_out, c, num_input_features):
        super(IndepSAB, self).__init__()
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.dim_out = dim_out
        self.c = c

        self.batch_size = num_input_features
        self.mabs = nn.ModuleList([
            MAB(self.dim_in, self.dim_in, self.dim_emb, self.dim_out, self.c)
            for _ in range(self.batch_size)])

    def forward(self, X):
        per_batch_outputs = []
        batch_split_X = X.split(1)

        for batch_index in range(self.batch_size):
            X_single = batch_split_X[batch_index]
            per_batch_outputs.append(
                self.mabs[batch_index](X_single, X_single))

        return torch.cat(per_batch_outputs)


class ISAB(nn.Module):
    """Multi-head Self-Attention Block with Inducing Points."""
    has_inducing_points = True

    def __init__(self, dim_in, dim_emb, dim_out, c,
                 num_inds, num_input_features=None):
        super(ISAB, self).__init__()
        # inducing points have dimensionality dim_out (i.e. dim_hidden)
        self.ind = nn.Parameter(torch.Tensor(1, num_inds, dim_emb))
        nn.init.xavier_uniform_(self.ind)

        # If we are performing additive or multiplicative encoding (see MAB)
        # we must provide dims for the attention matrices.
        if num_input_features is not None:
            mab0_dim_att = (num_inds, num_input_features)
            mab1_dim_att = (num_input_features, num_inds)
        else:
            mab0_dim_att = None
            mab1_dim_att = None

        # mab0 is MAB(ind, X)
        # i.e. query has dim_out, key and value (both X) have dim_in, and
        # we want to output dim_out
        self.mab0 = MAB(
            dim_emb, dim_in, dim_emb, dim_emb, c, dim_att=mab0_dim_att)

        # mab1 is between X, H
        # H is dim dim_out from mab
        # therefore we have dim_in=dim_in (from X), and H is already dim_out
        # and we also want to output dim_out
        self.mab1 = MAB(
            dim_in, dim_emb, dim_emb, dim_out, c, dim_att=mab1_dim_att)

    def forward(self, X):
        # repeat inducing points along batch dimension before applying mab0
        H = self.mab0(self.ind.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class IndepISAB(nn.Module):
    has_inducing_points = True

    def __init__(self, dim_in, dim_emb, dim_out, c,
                 num_inds, num_input_features=None):
        super(IndepISAB, self).__init__()
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.dim_out = dim_out
        self.c = c
        self.num_inds = num_inds
        self.batch_size = num_input_features

        # inducing points have dimensionality dim_out (i.e. dim_hidden)
        # Will likely throw a warning,
        # as per https://github.com/pytorch/pytorch/issues/49285
        self.inds = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, self.num_inds, self.dim_emb))
            for _ in range(self.batch_size)])

        for ind in self.inds:
            nn.init.xavier_uniform_(ind)

        # mab0 is MAB(ind, X)
        # i.e. query has dim_out, key and value (both X) have dim_in, and
        # we want to output dim_out
        self.mab0s = nn.ModuleList([
            MAB(self.dim_emb, self.dim_in, self.dim_emb, self.dim_emb, self.c)
            for _ in range(self.batch_size)])

        # mab1 is between X, H
        # H is dim dim_out from mab
        # therefore we have dim_in=dim_in (from X), and H is already dim_out
        # and we also want to output dim_out
        self.mab1s = nn.ModuleList([
            MAB(self.dim_in, self.dim_emb, self.dim_emb, self.dim_out, self.c)
            for _ in range(self.batch_size)])

    def forward(self, X):
        per_batch_outputs = []
        batch_split_X = X.split(1)

        for batch_index in range(self.batch_size):
            X_single = batch_split_X[batch_index]
            ind = self.inds[batch_index]
            mab0 = self.mab0s[batch_index]
            mab1 = self.mab1s[batch_index]

            # repeat inducing points along batch dimension before applying mab0
            H = mab0(ind, X_single)
            per_batch_outputs.append(mab1(X_single, H))

        return torch.cat(per_batch_outputs)


class IMAB(nn.Module):
    """Multi-head Self-Attention Block with Inducing Points.

    Simplified version of ISAB. No row interactions but additive loss."""
    has_inducing_points = True

    def __init__(self, dim_in, dim_emb, dim_out, c,
                 num_inds, num_input_features=None):
        super(IMAB, self).__init__()

        self.ind = nn.Parameter(torch.Tensor(1, num_inds, dim_emb))

        # If we are performing additive or multiplicative encoding (see MAB)
        # we must provide dims for the attention matrices.
        if num_input_features is not None:
            mab_dim_att = (num_input_features, num_inds)
        else:
            mab_dim_att = None

        nn.init.xavier_uniform_(self.ind)
        self.mab = MAB(
            dim_in, dim_emb, dim_emb, dim_out, c, dim_att=mab_dim_att)

    def forward(self, X):
        # repeat inducing points along batch dimension before applying mab
        return self.mab(X, self.ind.repeat(X.size(0), 1, 1))


class SlotMAB(nn.Module):
    """
    Similar to Slot Attention, but softmax over the values (not the queries),
        and not applied iteratively. MAB(I, X).
    """
    has_inducing_points = True

    def __init__(self, dim_in, dim_emb, dim_out, c,
                 num_inds, num_input_features=None):
        super(SlotMAB, self).__init__()

        self.ind = nn.Parameter(torch.Tensor(1, num_inds, dim_emb))

        # If we are performing additive or multiplicative encoding (see MAB)
        # we must provide dims for the attention matrices.
        if num_input_features is not None:
            mab_dim_att = (num_inds, num_input_features)
        else:
            mab_dim_att = None

        nn.init.xavier_uniform_(self.ind)
        self.mab = MAB(
            dim_emb,  # dim_Q
            dim_in,   # dim_KV
            dim_emb,  # dim_emb
            dim_out,  # dim_out
            c, dim_att=mab_dim_att)

    def forward(self, X):
        # repeat inducing points along batch dimension before applying mab
        return self.mab(self.ind.repeat(X.size(0), 1, 1), X)


class IndepIMAB(nn.Module):
    has_inducing_points = True

    def __init__(self, dim_in, dim_emb, dim_out, c,
                 num_inds, num_input_features=None):
        super(IndepIMAB, self).__init__()
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.dim_out = dim_out
        self.c = c
        self.num_inds = num_inds

        self.batch_size = num_input_features

        # Will likely throw a warning,
        # as per https://github.com/pytorch/pytorch/issues/49285
        self.inds = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, self.num_inds, self.dim_emb))
            for _ in range(self.batch_size)])

        for ind in self.inds:
            nn.init.xavier_uniform_(ind)

        self.mabs = nn.ModuleList([
            MAB(self.dim_in, self.dim_emb, self.dim_emb, self.dim_out, self.c)
            for _ in range(self.batch_size)])

    def forward(self, X):
        per_batch_outputs = []
        batch_split_X = X.split(1)

        for batch_index in range(self.batch_size):
            X_single = batch_split_X[batch_index]
            ind = self.inds[batch_index]
            mab = self.mabs[batch_index]

            # repeat inducing points along batch dimension before applying mab
            per_batch_outputs.append(mab(X_single, ind))

        return torch.cat(per_batch_outputs)


class Prototypes(nn.Module):
    """Prototypes.

    These are just nn.Parameter's which are concatenated to the input X
    to the AttentionBlocks alongside the row dimension of the table.

    The values of the prototypes are globally learned using gradient descent.

    Since they are processed similar as other input, we refer to them as
    prototypes.
    """
    def __init__(self, num_prototypes, num_features, dim_embedding):
        """Init Prototypes.

        (num_features, dim_embedding) have previously been referred to as
        (D, E) in other docstrings.

        """
        super().__init__()
        self.num = num_prototypes
        prototype_shape = (num_prototypes, num_features, dim_embedding)
        self.prototypes = nn.Parameter(torch.Tensor(*prototype_shape))
        nn.init.xavier_uniform_(self.prototypes)

    def concat(self, X):
        """Concatenate prototypes to input tensor X."""
        return torch.cat([X, self.prototypes], 0)

    def discard(self, X):
        """Remove prototypes from input tensor X.

        Assumes prototypes have been added with concat()
        """
        return X[:-self.num]
