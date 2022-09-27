"""Contains Tabular Transformer Model definition."""
import re
from functools import partial
from itertools import cycle
from typing import List

import torch
import torch.nn as nn
from memory_profiler import profile

from ttnn.model.image_patcher import (
    LinearImagePatcher, ConvImagePatcher, SlotMABImagePatcher)
from ttnn.model.multitask_uncertainty import MultitaskUncertainty
from ttnn.model.supcon_loss import SupConLoss
from ttnn.model.ttnn_modules import (
    MAB, SAB, ISAB, IMAB, Prototypes, IndepSAB, IndepIMAB, IndepISAB, SlotMAB)
from ttnn.utils.config_utils import Args
from ttnn.utils.encode_utils import torch_cast_to_dtype
from typing import Optional
from ttnn.utils.param_init_utils import xavier_normal_identity_
import numpy as np
from ttnn.model.hpc_modules import Hierarchical_Perceiver

ATT_BLOCK_NAME_TO_CLASS = {
    'ISAB': ISAB,
    'SAB': SAB,
    'IMAB': IMAB,
    'IndepISAB': IndepISAB,
    'IndepSAB': IndepSAB,
    'IndepIMAB': IndepIMAB,
    'SlotMAB': SlotMAB
}

IMAGE_PATCHER_SETTING_TO_CLASS = {
    'linear': LinearImagePatcher,
    'conv': ConvImagePatcher,
    'slot-mab': SlotMABImagePatcher
}


def trunc_normal_(x, mean=0., std=1.):
    """Truncated normal initialization (approximation)."""
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-
    # initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class rFF(nn.Module):
    def __init__(self, dim_in, dim_out, c):
        super().__init__()
        hidden_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)
        rff_depth = c.model_rff_depth

        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        rff = [nn.Linear(dim_in, 4 * dim_out), nn.GELU()]

        if hidden_dropout is not None:
            rff.append(hidden_dropout)

        for i in range(rff_depth - 1):
            rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if hidden_dropout is not None:
                rff.append(hidden_dropout)

        rff += [nn.Linear(4 * dim_out, dim_out)]

        if hidden_dropout is not None:
            rff.append(hidden_dropout)

        self.rFF = nn.Sequential(*rff)

    def forward(self, X):
        return self.rFF(X)


class TTNNModel(nn.Module):
    """Tabular transformer neural networks.

    Applies Multi-head self-attention blocks to tabular data.

    There are two main flavours of TTNN, which we call `nested` and
    `flattened` TTNN.
    The 'flattened' TTNN conducts only Attention Between Datapoints operations,
    while the 'nested' TTNN conducts ABD followed by ABA
    The docstring of `__init__()` describes the other model configuration
    options in detail.

    For all model variants, we expect a list of input data, `X_ragged`:
    ```
        len(X_ragged) == N
        X_ragged[i].shape == (D, H_i)
    ```
    In other words, we have `N` input samples. All samples share the same
    number of `D` features, where each feature i is encoded in `H_i`
    dimensions. "Encoding" here refers to the data preprocessing, i.e. the
    one-hot-encoding for categorical features, as well as well as adding
    the mask tokens. (Note that this is done by the code and the user is
    expected to provide datasets as given in `ttnn.data_loaders`.)

    High-level model overview:

    Initially in TTNNModel, `self.in_embedding()` linearly embeds each of the
    `D` feature columns to a shared embedding dimension `E`.
    We learn separate embedding weights for each column.
    This allows us to obtain the embedded data matrix `X_emb` as a
    three-dimensional tensor of shape `(N, D, E)`.
    `E` is referred to as `dim_feat` in the code below.

    After embedding the data, we either apply flattened or nested TTNN.
    See `build_flattened_ttnn()` and `build_nested_ttnn()` for further
    information on these. Both models essentially apply a series of attention
    blocks on the input.

    For both variants, we eventually obtain output of shape `(N, D, E)`,
    which is projected back to the dimensions of the input `X_ragged` using
    `self.out_embedding()`, which applies a learned linear embedding to
    each column `D` separately.

    """
    def __init__(
            self, c, metadata, device=None,
            supcon_target_col=None):
        """Initialise TTNNModel.

        Args:
            c: wandb config
            metadata: Dict, from which we retrieve:
                input_feature_dims: List[int], used to specify the number of
                    one-hot encoded dimensions for each feature in the table
                    (used when reloading models from checkpoints).
                cat_features: List[int], indices of categorical features, used
                    in model initialization if using feature type embeddings.
                num_features: List[int], indices of numerical features, used
                    in model initialization if using feature type embeddings.
                cat_target_cols: List[int], indices of categorical target
                    columns, used if there is a special embedding dimension
                    for target cols.
                num_target_cols: List[int], indices of numerical target
                    columns, used if there is a special embedding dimension
                    for target cols.
                time_code: List[List[int]], indices indicating which data
                    belongs to which time step. First index of list is static
                    data. Remaining indices correspond to each timestep. 
            device: Optional[int], must be specified for MultitaskUncertainty.
            supcon_target_col: Optional[int], specifies the index of the target
             column, if the user aims to use Supervised Contrastive Loss
            (Khosla et al. 2020, https://arxiv.org/pdf/2004.11362.pdf)

        The following different configurations are available.

        ** (1) SAB / ISAB / IMAB **

        Config argument: `c.model_att_block` (str in {'SAB', ISAB', 'IMAB'})
        This refers to the flavour of attention blocks used. SAB and ISAB are
        identical to how they are defined in the Set Transformer paper. IMAB
        is an attention block architecture of our creation, for which, unlike
        for SAB and ISAB, the loss decomposes over rows. Therefore, IMAB is
        compatible with mini-batch objectives / data sub-sampling.

        If MAB is a multihead attention block, then

        * SAB(X) = MAB(X, X, X)

        * ISAB(X) = MAB(X, H, H)
          H = MAB(I, X, X)

        * IMAB(X) = MAB(X, I, I).

        ** (2) Prototypical Rows ***

        Config argument: c.model_prototypes (int)
        A number of `c.model_prototypes` globally learned parameters may be
        concatenated to the embedded input matrix `X` prior to application
        of the attention block sequence.
        These parameters can be interpreted as protoypical rows. Their
        values are learned as global parameters with gradient descent.

        ** (3) Multitask Uncertainties **

        Learn the uncertainties for each feature column as global parameters.
        See `ttnn.models.multitask_uncertainty` for full details.

        """
        super().__init__()

        # *** Extract Configs ***
        # cannot deepcopy wandb config.
        if c.mp_distributed:
            self.c = Args(c.__dict__)
        else:
            self.c = Args(c)

        # * Main model configuration *
        self.model_type = c.model_type
        self.is_flattened = c.is_flattened
        self.has_column_attention = c.has_column_attention
        self.model_nested_row_att_block = c.model_nested_row_att_block
        self.device = device
        self.model_recurrence_interval = c.model_recurrence_interval
        self.model_recurrence_share_weights = c.model_recurrence_share_weights

        # * Dataset Metadata *
        input_feature_dims = metadata['input_feature_dims']
        cat_features = metadata['cat_features']
        num_features = metadata['num_features']
        cat_target_cols = metadata['cat_target_cols']
        num_target_cols = metadata['num_target_cols']

        if 'time_code' in metadata:
            self.time_code = metadata['time_code']
        else:
            self.time_code = None

        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = c.model_stacking_depth

        # each feature column gets embedded to dim_feature_embedding.
        dim_feature_embedding = c.model_dim_feat_embedding

        # # if specified, target columns (numerical or categorical) are
        # # embedded to this size
        # dim_target_embedding = c.model_dim_target_embedding

        # the dimension of the output of multihead attention is given by
        self.dim_hidden = c.model_dim_hidden
        # dim_hidden is spread across num_heads heads
        self.num_heads = c.model_num_heads

        assert (self.dim_hidden % self.num_heads == 0), (
            f"'num_heads' needs to divide 'dim_hidden' without remainder "
            f"because this 'dim_hidden' gets spread across the heads.")

        # how many feature columns are in the input data
        # apply image patching if specified
        if self.c.model_image_n_patches:
            if self.c.model_image_patch_type == 'slot-mab':
                if device is None:
                    raise Exception(
                        'Must provide device for slot att patching.')
                extra_args = {'device': device}
            else:
                extra_args = {}

            # num_input_features = n_patches per image
            self.image_patcher = IMAGE_PATCHER_SETTING_TO_CLASS[
                self.c.model_image_patch_type](
                input_feature_dims=input_feature_dims,
                dim_feature_embedding=dim_feature_embedding,
                dim_hidden=self.dim_hidden,
                c=c, **extra_args)
            ttnn_attrs = self.image_patcher.get_ttnn_attrs()
            for k, v in ttnn_attrs.items():
                self.__setattr__(name=k, value=v)

            # self.add_module('ImagePatchEncoder', self.image_patcher)
        else:
            self.image_patcher = None
            self.num_input_features = len(input_feature_dims)

        # supervised contrastive regularizer module
        # only valid for production, single target classification
        if self.c.model_supcon_regularizer:
            self.supcon_regularizer = SupConLoss(c)
            self.supcon_target_col = supcon_target_col
        else:
            self.supcon_regularizer = None

        # TODO: for any data modality that permits data augmentation, should
        #   be able to use self-supervised contrastive regularizer.

        # whether or not to add a feature type embedding
        self.use_feature_type_embedding = c.model_feature_type_embedding

        # whether or not to add a feature index embedding
        self.use_feature_index_embedding = c.model_feature_index_embedding

        # *** Set up the attention blocks to be used. ***

        try:
            self.att_block = ATT_BLOCK_NAME_TO_CLASS[c.model_att_block]
        except KeyError as e:
            raise NotImplementedError(e)

        # *** Build Model ***

        # Flattened model
        if self.is_flattened or self.c.model_type == 'hybrid-inducing':
            if self.c.model_use_pre_npt_rff:
                self.rff_dim_input = np.sum(input_feature_dims)
                print(f'Using rFF embedding with '
                      f'{self.rff_dim_input} input dims')

            self.dim_input = (dim_feature_embedding * self.num_input_features)
            print(f'hidden dimension expanded to {self.dim_input} input dims')

        else:
            if dim_feature_embedding != self.dim_hidden:
                print(
                    f'Using a non-flat TTNN model. '
                    f'Setting dim_feature_embedding (prev: '
                    f'{dim_feature_embedding}) to dim_hidden '
                    f'({self.dim_hidden}).')
            dim_feature_embedding = self.dim_hidden

        # For Nested and Hybrid model, we will immediately embed each element
        # (i.e., a table with N rows and D columns has N x D elements)
        # to the hidden_dim. Similarly, in the output, we will "de-embed"
        # from this hidden_dim.
        # This simplifies implementation for Nested and Hybrid, and allows
        # weight sharing in the recurrent architecture throughout the network.

        # Make as close to dim_in as possible,
        # otherwise is 0 for dim_in > num_heads
        # self.last_embed = max(
        #     self.num_heads,
        #     self.dim_input // self.num_heads * self.num_heads)

        # Build encoder
        model_type_to_enc_builder = {
            'flattened': self.get_flattened_ttnn,
            'hybrid': self.get_hybrid_ttnn,  # hybrid is working
            'hybrid-inducing': self.get_hybrid_ttnn,
            'nested': self.get_nested_ttnn,   # nested is not working
            'hybrid-custom': self.get_hybrid_custom_stack_ttnn,
            'i-npt': self.get_flattened_inducing_npt,
            'iterative-inducing': self.get_iterative_inducing_npt,
            'h-npt-large': self.get_hierarchical_i_npt,
        }
        if c.model_ablate_col_attention_only:
            self.enc = self.get_column_only_ttnn()
        else:
            self.enc = model_type_to_enc_builder[c.model_type]()

        # *** Dropout and LayerNorm in In-/Out-Embedding ***

        # Hidden dropout is applied for in- and out-embedding
        self.embedding_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        # LayerNorm applied after embedding, before dropout
        if self.c.embedding_layer_norm and device is None:
            print(
                'Must provide a device in TTNN initialization with embedding '
                'LayerNorm.')
        elif self.c.embedding_layer_norm:
            if self.has_column_attention:
                # For nested/hybrid TTNN, we batch over rows and columns
                # (i.e. just normalize over E)
                layer_norm_dims = [dim_feature_embedding]
            else:
                if self.c.model_use_pre_npt_rff:
                    in_layer_norm_dims = [self.rff_dim_input]
                    self.in_embedding_layer_norm = nn.LayerNorm(
                        in_layer_norm_dims, eps=self.c.model_layer_norm_eps)

                # For flattened TTNN, we batch over rows, and normalize
                # over the (D x E)
                layer_norm_dims = [
                    self.num_input_features * dim_feature_embedding]

            self.embedding_layer_norm = nn.LayerNorm(
                layer_norm_dims, eps=self.c.model_layer_norm_eps)

        else:
            self.in_embedding_layer_norm = None
            self.embedding_layer_norm = None

        # *** Input In/Out Embeddings ***
        # Don't use for Image Patching - those are handled by the respective
        # init_image_patching

        # In-Embedding
        # Linearly embeds each of the `D` [len(input_feature_dims)] feature
        # columns to a shared embedding dimension E [dim_feature_embedding].
        # Before the embedding, each column has its own dimensionionality
        # H_j [dim_feature_encoding], given by the encoding dimension of the
        # feature (e.g. This is given by the one-hot-encoding size for
        # categorical variables + one dimension for the mask token and two-
        # dimensional for continuous variables (scalar + mask_token)).
        # See docstring of TTNNModel for further context.

        # TODO: can we have a separate hidden dims for each column in nested?
        #   Probably not, how would we weight-share?
        #   Can we for flattened? Yes, probably.
        # if dim_target_embedding is None:
        #     # Same feature embedding size for all features
        #     dim_feature_embeddings = [
        #         dim_feature_embedding] * len(input_feature_dims)
        # else:
        #     # Use a special embedding size for the target(s)
        #     dim_feature_embeddings = []
        #     target_cols = set(cat_target_cols + num_target_cols)
        #     for feature_index in range(len(input_feature_dims)):
        #         if feature_index in target_cols:
        #             dim_feature_embeddings.append(dim_target_embedding)
        #         else:
        #             dim_feature_embeddings.append(dim_feature_embedding)

        if self.image_patcher is None:
            if self.c.model_use_pre_npt_rff:
                # We assume that the model is flattened
                # Apply an rFF across all the columns
                # TODO(nband): add a ResNet encoder
                pass
            else:
                if self.c.model_complex_in_embedding:
                    # Use 2 Layer MLP to handle in-embeddings
                    self.in_embedding = nn.ModuleList([
                        nn.Sequential(*[
                            nn.Linear(
                                dim_feature_encoding,
                                2*dim_feature_embedding),
                            nn.GELU(),
                            nn.Linear(
                                2*dim_feature_embedding,
                                dim_feature_embedding)])
                        for dim_feature_encoding in input_feature_dims])
                elif self.time_code:
                    self.in_embedding = nn.ModuleList([
                        nn.Linear(dim_feature_encoding, dim_feature_embedding)
                        for dim_feature_encoding in input_feature_dims[:self.time_code[1][-1]+1]])
                else:
                    self.in_embedding = nn.ModuleList([
                        nn.Linear(dim_feature_encoding, dim_feature_embedding)
                        for dim_feature_encoding in input_feature_dims])

        # Feature Type Embedding
        # Optionally, we construct "feature type" embeddings -- i.e. we learn a
        # representation based on whether the feature is either
        # (i) numerical or (ii) categorical.
        if self.use_feature_type_embedding:
            if cat_features is None or num_features is None:
                raise Exception(
                    'Must provide cat_feature and num_feature indices at '
                    'TTNN initialization if you aim to compute feature type'
                    ' embeddings.')

            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to TTNN initialization: in '
                    'distributed setting, and aim to do feature type '
                    'embedding.')

            # If all features are either categorical or numerical,
            # don't bother.
            if len(cat_features) == 0 or len(num_features) == 0:
                print(
                    'All features are either categorical or numerical. '
                    'Not going to bother doing feature type embeddings.')
                self.feature_type_embedding = None
            else:
                self.feature_types = torch_cast_to_dtype(torch.empty(
                    self.num_input_features, device=device), 'long')

                for feature_index in range(self.num_input_features):
                    if feature_index in num_features:
                        self.feature_types[feature_index] = 0
                    elif feature_index in cat_features:
                        self.feature_types[feature_index] = 1
                    else:
                        raise Exception

                self.feature_type_embedding = nn.Embedding(
                    2, dim_feature_embedding)

            print(
                f'Using feature type embedding (unique embedding for '
                f'categorical and numerical features).')
        else:
            self.feature_type_embedding = None

        # Feature Index Embedding
        # Optionally, learn a representation based on the index of the column.
        # Allows us to explicitly encode column identity, as opposed to
        # producing this indirectly through the per-column feature embeddings.
        if self.use_feature_index_embedding:
            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to TTNN initialization: in '
                    'distributed setting, and aim to do feature index '
                    'embedding.')
            if self.time_code:
                self.feature_indices = torch_cast_to_dtype(
                    torch.tensor(
                        sum([[i]*len(d) for i, d in enumerate(
                            self.time_code)], []), device=device), 'long')

                self.feature_index_embedding = nn.Embedding(
                    len(self.time_code),
                    dim_feature_embedding)
                print(
                    f'Using non-unique feature embedding which'
                    f'repeats each timestep.')
            else:
                self.feature_indices = torch_cast_to_dtype(
                    torch.arange(
                        self.num_input_features, device=device), 'long')

                self.feature_index_embedding = nn.Embedding(
                    self.num_input_features, dim_feature_embedding)

                print(
                    f'Using feature index embedding (unique embedding for '
                    f'each column).')
        else:
            self.feature_index_embedding = None

        # Out embedding.
        # The outputs of the AttentionBlocks have shape (D, E, H)
        # [len(input_feature_dim), dim_feature_embedding, dim_hidden].
        # (For nested TTNN this arises naturally, for flattened, we reshape
        # from (D*E, H)).
        # For each of the column j, we then project back to the dimensionality
        # of that column in the input (N, H_j-1), subtracting 1, because we do
        # not predict the mask tokens, which were present in the input.

        if self.image_patcher is None:

            # Need to remove the mask column if we are using BERT augmentation,
            # otherwise we just project to the same size as the input.
            if self.c.model_bert_augmentation:
                get_dim_feature_out = lambda x: x - 1
            else:
                get_dim_feature_out = lambda x: x

            if self.time_code:
                self.out_embedding = nn.ModuleList([
                        nn.Linear(
                            dim_feature_embedding,
                            get_dim_feature_out(dim_feature_encoding))
                        for dim_feature_encoding in input_feature_dims[:self.time_code[1][-1]+1]])
            else:
                self.out_embedding = nn.ModuleList([
                    nn.Linear(
                        dim_feature_embedding,
                        get_dim_feature_out(dim_feature_encoding))
                    for dim_feature_encoding in input_feature_dims])

        # *** Learn Prototypical Rows ***
        if c.model_prototypes > 0:
            self.prototypes = Prototypes(
                c.model_prototypes,
                self.num_input_features,
                dim_feature_embedding)
        else:
            self.prototypes = None

        # *** Learn Multitask Uncertainties ***
        if c.model_multitask_uncertainties:
            if device is None:
                raise Exception('Must specify device on TTNN initialization '
                                'for use with MultitaskUncertainty.')
            # Note that we specify len(input_feature_dims) instead of
            # self.num_input_features, because even if we apply image
            # patching, this is applied after we have projected back to the
            # flattened image size (in particular, Multitask Uncertainty
            # is applied in the loss)
            self.uncertainties = MultitaskUncertainty(
                len(input_feature_dims)).to(device=device)
        else:
            self.uncertainties = None

        # *** Gradient Clipping ***
        if c.exp_gradient_clipping:
            clip_value = c.exp_gradient_clipping
            print(f'Clipping gradients to value {clip_value}.')
            for p in self.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))

        # param_names = []
        # for n, _ in self.named_parameters():
        #     param_names.append(n)

        # *** T-Fixup Initialization ***
        # From `Improving Transformer Optimization Through Better
        # Initialization` (Huang et al. 2020)
        #   * We suggest using T-Fixup with LayerNorm disabled and
        #   * no learning rate warmup.
        if c.model_tfixup:
            # Implementation adapted from the 3rd place solution to
            # Riiid! Answer Correctness Prediction Kaggle competition.
            # https://github.com/jamarju/riiid-acp-pub

            # See also fairseq implementation of TFixup -
            # https://github.com/layer6ai-labs/T-Fixup/blob/f1fae213ce7b48829
            # f81632d0c96bb039b7c450e/fairseq/modules/multihead_attention.py
            for n, p in self.named_parameters():
                # No special init for biases, BatchNorm / LayerNorm weights,
                # multitask_uncertainty weights if we are using them
                pattern = r'.*bias$|.*bn\.weight$|enc.*.ln.*.weight'
                pattern += r'|uncertainties.*|.*att.*encoding'
                pattern += r'|.*conv.*'
                pattern += r'|image_patcher.*.ln.*.weight'
                if re.match(pattern, n):
                    continue

                gain = 1

                # T-Fixup attention-specific initialization:
                # Scale the V and mix_head matrices used in self-attention,
                # and the weight matrices used by the row-wise feed-forward
                # networks (rFFs)
                if re.match(
                        r'enc.*.fc_v.weight|enc.*.fc_mix_heads.weight|'
                        r'enc.*.rff.*.weight', n):
                    gain = 0.67 * (c.model_stacking_depth ** (-1. / 4.))
                    if re.match(r'enc.*.fc_v.weight', n):
                        gain *= (2 ** 0.5)
                if re.match(r'^in_embedding', n):
                    # Use Gaussian initialization N(0; d^{-1/2}) for input
                    # embeddings where d is the embedding dimension.
                    trunc_normal_(
                        p.data,
                        std=(4.5 * c.model_stacking_depth) ** (
                            -1. / 4.) * c.model_dim_hidden ** (-0.5))
                else:
                    nn.init.xavier_normal_(p, gain=gain)

    # Reinitialize all fc_q, fc_k, fc_v weights to be centered around
    # the identity.
        if c.model_qkv_embedding_identity_init:
            if c.model_share_qk_sab_embedding:
                re_string = r'enc.*.fc_qk.weight'
            else:
                re_string = (
                    r'enc.*.fc_q.weight|enc.*.fc_k.weight|enc.*.fc_v.weight')

            gain = 1
            for n, p in self.named_parameters():
                if re.match(re_string, n):
                    xavier_normal_identity_(p, gain=gain)

            print(
                f'Built TTNN model with base attention block '
                f'{c.model_att_block}, model type '
                f'{c.model_type}.')

    def get_flattened_ttnn(self):
        """Flattened TTNN.

        Performs attention between the rows `N`.
        We stack a sequence of the chosen AttentionBlock (ISAB, SAB, or IMAB)
        until we have reached the total `stacking_depth`.

        The `AttentionBlock` expects input of shape (B_A, N_A, F_A), where
        `B_A` is a batch dimension, along which we parallelise computation,
        `N_A` is the number of samples in each batch, along which we perform
        attention, and `F_A` is the dimension of feature embedding.
        (`F_A` is the `dim_in := num_input_features * dim_feature_embedding`
        for the first AttentionBlock and `dim_hidden` for all
        consequent blocks.)

        To perform attention between the rows of our input, for flattened TTNN
        we reshape our input from shape (N, D, E) to
        (1, N, D * E) = (B_A, N_A, F_A).
        (This happens in `forward()`.)

        NOTE: Strictly speaking, we could choose a separate embedding dimension
        for each feature, if we are using the flattened TTNN. This is currently
        not supported in the code, but may be required to save computation for
        large datasets. The input size would then be (1, N, sum_j E_j), where
        E_j is the embedding dimension for each column j in D.

        The first `AttentionBlock` has input data of shape (1, N, D * E)
        and outputs data of shape (1, N, dim_hidden).

        Below, `D*E` corresponds to
        `dim_input = num_input_features * dim_feature_embedding`.

        In *each* `AttentionBlock`, *for each head*, the query, key, and
        value matrices are embedded to a shared dimension
        `dim_hidden // num_heads`, where `num_heads` is the number of self-
        attention heads.
        The embedding are not shared between the heads.

        NOTE: In the code this is implemented as a global embedding
        `Q = self.fc_q(X)` with shape (N_A, dim_hidden), which is then split
        along `dim_hidden` into `num_heads` different queries. This is
        equivalent to what we have described above.

        NOTE: This implementation of the multihead attention block
        is identical to the Set Transformer code, but diverges from what is
        stated in the Set Transformer publication, which claims that the output
        dimension of MultiHeadAttention is identical to the input dimension.
        This is not the case here, as we / they choose the dimension of the
        output to be `dim_hidden`.
        For further info see the following Github issue:
        https://github.com/juho-lee/set_transformer/issues/8.

        Subsequent `AttentionBlock`s have inputs of shape
        (1, N, dim_hidden) and output tensors of shape (1, N, dim_hidden).
        The last `AttentionBlock` has input of shape (1, N, dim_hidden)
        and outputs shape (1, N, D * E).

        NOTE: I have just changed this behaviour. The last `AttentionBlock`
        has previously output shape (1, N, dim_hidden), which was then linearly
        transformed to shape (1, N, D * E). I think it's better to let the
        transformer handle the scaling to the original dimension.

        In `forward()`, we then reshape the output to (N, D, E),
        split across `D` to obtain `D` arrays of shape (N, E),
        and use `self.out_embedding` to project back to the dimensions
        of `X_ragged`, the input dimensions (where we subtract 1 from
        the feature encoding dimensions, because we currently do not want to
        predict mask tokens).
        """
        args = dict(c=self.c)
        if getattr(self.att_block, 'has_inducing_points', False):
            args['num_inds'] = self.c.model_num_inds

        if self.stacking_depth == 1:
            return self.att_block(
                self.dim_input, self.last_embed, self.dim_input, **args)

        enc = [
            self.att_block(
                self.dim_input, self.dim_hidden, self.dim_hidden, **args)]

        for i in range(1, self.stacking_depth):
            enc.append(self.att_block(
                self.dim_hidden, self.dim_hidden, self.dim_hidden, **args))

        # # We cannot project back to dim_input directly, because
        # # dim input may not be divisible by num_heads
        # # so we project back to closest number dim_out that is
        # enc.append(self.att_block(
        #     self.dim_hidden, self.last_embed, self.dim_input, **args))

        # Add a final linear transform from the hidden dimension to the
        # input dimension
        enc.append(nn.Linear(self.dim_hidden, self.dim_input))

        enc = nn.Sequential(*enc)
        return enc

    def get_flattened_inducing_npt(self):
        """
        This implements a model rougly equivalent to the PerceiverIO
        (see https://arxiv.org/abs/2107.14795).

        First apply cross-attention
        MAB(I, X)
        to obtain an intermediate representation of dimensions
        M x (D * E)

        Then repeatedly apply self-attention in this smaller
        space, eventually obtaining final inducing representation Z.

        Finally, apply cross-attention to project to output dims.
        MAB(X, Z)
        """
        args = dict(c=self.c)

        if self.stacking_depth == 1:
            raise NotImplementedError

        if self.c.model_use_pre_npt_rff:
            enc = [
                rFF(self.rff_dim_input, self.dim_hidden, self.c),
                SlotMAB(self.dim_hidden, self.dim_hidden, self.dim_hidden,
                        self.c,
                        self.c.model_num_inds)]
        else:
            # MAB(I, X)
            enc = [
                SlotMAB(self.dim_input, self.dim_hidden, self.dim_hidden,
                        self.c, self.c.model_num_inds)]

        # Repeated SABs
        for i in range(1, self.stacking_depth - 1):
            enc.append(SAB(
                self.dim_hidden, self.dim_hidden, self.dim_hidden, **args))

        enc = nn.Sequential(*enc)

        # MAB(X, Z)
        if self.c.model_use_pre_npt_rff:
            self.final_mab = MAB(
                self.rff_dim_input, self.dim_hidden, self.dim_hidden,
                self.dim_hidden, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)
        else:
            self.final_mab = MAB(
                self.dim_input, self.dim_hidden, self.dim_hidden,
                self.dim_hidden, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)

        self.final_linear = nn.Linear(self.dim_hidden, self.dim_input)

        return enc

    def get_iterative_inducing_npt(self):
        """
        cross-attention is applied first in the batch
        and then in the feature dimensions
        """
        args = dict(c=self.c)
        D = self.num_input_features

        if self.stacking_depth == 1:
            raise NotImplementedError

        if self.c.model_use_pre_npt_rff:
            enc = [
                rFF(self.rff_dim_input, self.dim_hidden * D, self.c),
                SlotMAB(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D, self.c, self.c.model_num_inds)]
        else:
            enc = [
                SlotMAB(
                    self.dim_input, self.dim_hidden * D,
                    self.dim_hidden * D, self.c, self.c.model_num_inds)]

        for i in range(1, self.stacking_depth - 1):
            enc.append(SAB(
                self.dim_hidden * D, self.dim_hidden * D, self.dim_hidden * D,
                **args))

        enc.append(ReshapeToNested(D=D))
        enc = nn.Sequential(*enc)

        attribute_enc = [SlotMAB(
            self.dim_hidden, self.dim_hidden, self.dim_hidden,
            self.c, self.c.model_num_inds)]

        for i in range(1, self.stacking_depth - 1):
            attribute_enc.append(SAB(
                self.dim_hidden, self.dim_hidden,
                self.dim_hidden, **args))

        self.attribute_enc = nn.Sequential(*attribute_enc)

        self.recover_attr_MAB = MAB(
                self.dim_hidden, self.dim_hidden,
                self.dim_hidden, self.dim_hidden,
                self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)
        # MAB(X, Z)
        if self.c.model_use_pre_npt_rff:
            self.recover_batch_MAB = MAB(
                self.rff_dim_input, self.dim_hidden * D, self.dim_hidden * D,
                self.dim_hidden * D, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)
        else:
            self.recover_batch_MAB = MAB(
                self.dim_input, self.dim_hidden * D, self.dim_hidden * D,
                self.dim_hidden * D, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)

        self.final_flatten = ReshapeToFlat()
        self.final_unflatten = ReshapeToNested(D=D)
        self.final_linear = nn.Linear(self.dim_hidden * D, self.dim_input)
        return enc

    def get_nested_ttnn(self):
        """Nested TTNN.

        Nested TTNN proceeds along similar lines as Flattened TTTN. It is
        suggested you read the docstring `get_flattened_ttnn()` first.

        Again, we stack a sequence of the chosen AttentionBlock (ISAB, SAB,
        IMAB) until we have reached the total `stacking_depth`.

        Again, `AttentionBlock` expects input of shape (B_A, N_A, F_A), where
        `B_A` is a batch dimension, along which we parallelise computation,
        `N_A` is the attention dimension, and `F_A` is the size of the
        embedding for each sample.

        **Nested TTNN now performs attention alternatingly between the columns
        `D` and rows `N` of the dataset.**

        That is, initially our data has shape (N, D, E). Unlike for the
        flattened Transformer, we now actually want to treat the N-
        dimension of this tensor as a batch dimension, across which we share
        computations.
        We perform the following operations (A-B) alternatingly:

        (A) Row-Attention:
        Nested TTNN first permutes the data tensor dimensions to be
        (D, N, E) and then performs attention between the rows `N`.
        This computation is executed in parallel and independently across
        feature columns `D`. Thus this operation compares data across
        rows/instances but not across columns/features.

        (B) Col-Attention:
        We then permute the tensor to be of shape (N, D, E) and perform
        attention over the columns `D`, in parallel and independently across
        rows `N`. This operation compares data across columns/features but
        not across rows/instances.

        NOTE: Currently `stacking_depth` counts each `AttentionBlock` in
        the nested TTNN. I.e. one full round (A-B) of nested TTNN corresponds
        to depth 2. Higher depths are most likely required for nested TTNN,
        since we need a full round of (A-B) to pass information between
        different columns and different rows.

        For nested TTNN, the `input_dim` (the last dimension of the tensor)
        is just given by `E`, the shared embedding dimension across columns.
        (Note that each column learns its own embedding matrix.)

        Again, in *each* `AttentionBlock` the input is (B_A, N_A, F_A),
        and for *for each head*, the query, key, and value matrices (N_A, F_A)
        are embedded to a shared dimension `dim_hidden // num_heads`,
        where `num_heads` is the number of self-attention heads.
        (B_A, N_A) now alternatingly corresponds to (N, D) (Col-Attention)
        and (D, N) (Row-Attention).

        Speculation: In general, the existence of a batch dimension should not
        mess with any in- or equi-variance properties of the model. (As long
        as we do full batch gradient descent.) I have to think about this some
        more. But the model should be equivariant w.r.t. everything in the
        batch dim. Therefore, nested TTNN should have the same row-equivariance
        properties as flat TTNN and additionally be equivariant w.r.t. columns.
        Also, if we use IMAB for the Row-Attention, our loss should decompose
        row-wise (mini-batching), regardless of the type of `AttentionBlock`
        used in Col-Attention.

        NOTE: Currently we share the `AttentionBlock`, as well as the
        `dim_hidden` between Col- and Row-Attention. It may be advantageous to
        not do that. (E.g., in order to implement IMAB for Row-Attention
        and SAB/ISAB for Col-Attention.)

        The first `AttentionBlock` has input (D, N, E) and outputs
        (D, N, dim_hidden).
        Subsequent `AttentionBlock`s have inputs of shape
        (..., dim_hidden) and output tensors of shape (..., dim_hidden).
        (Where `...` signifies (D, N) and (N, D) alternatingly).)
        The last `AttentionBlock` has input of shape (..., dim_hidden)
        and outputs (..., E).

        NOTE: I have just changed this behaviour. The last `AttentionBlock`
        has previously output shape (..., dim_hidden), which was then linearly
        transformed to shape (..., E). I think it's better to let the
        transformer handle the upscaling.

        Finally, if `stacking_depth` is even, we need to reshape the output
        from shape (D, N, E) to (N, D, E). (This is handled in the `forward()`)
        If `stacking_depth` is even, the output is already shape (N, D, E).

        We then split the output across `D` to obtain `D` arrays of
        shape (N, E), and use `self.out_embedding` to project
        back to the dimensions of X_ragged, the input dimensions.
        (Where we subtract 1 from the feature encoding dimensions,
        because we currently do not want to predict mask tokens.)

        self.model_nested_row_att_block specifies the AttentionBlock used for
        attention over the rows. The AttentionBlock for column-wise attention
        is still given by c.model_att_block.

        The user should specify self.model_nested_row_att_block = IMAB to make
        the nested TTNN compatible with mini-batching.
        """
        if self.stacking_depth < 2:
            raise Exception(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')

        # Retrieve the specified AttentionBlock for attention over the rows
        try:
            row_attention_block = ATT_BLOCK_NAME_TO_CLASS[
                self.model_nested_row_att_block]
        except KeyError:
            raise NotImplementedError

        print('Building nested transformer.')
        print(f'Row-wise attention block: {row_attention_block.__name__}')
        print(f'Column-wise attention block: {self.att_block.__name__}')
        rff_type = 'Gated GeLU' if self.c.model_rff_gated_gelu else 'GeLU'
        print(f'Row-wise feed-forward activation: {rff_type}')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'c': self.c}
        col_att_args = {'c': self.c}

        # Get correct number of inducing points for attention blocks.
        # There is the option to specify a different number of inducing
        # points for row and column attention.
        if getattr(self.att_block, 'has_inducing_points', False):
            col_att_args['num_inds'] = self.c.model_num_inds
        if getattr(row_attention_block, 'has_inducing_points', False):
            if (row_inds := self.c.model_num_row_inds) != -1:
                row_att_args['num_inds'] = row_inds
            else:
                row_att_args['num_inds'] = self.c.model_num_inds

        # Provide the number of columns to the column attention block when
        # additive or multiplicative encodings are enabled. These are
        # applied in the attention module, before and after the softmax
        # is applied, respectively. See ttnn_modules.py.
        if (self.c.model_att_additive_encoding or
                self.c.model_att_multiplicative_encoding):
            col_att_args['num_input_features'] = self.num_input_features

        # Provide number of columns to Indep attention block, which is used
        # for attention over the rows (i.e. an indepedent attention map is
        # learned for each column).
        if 'Indep' in self.model_nested_row_att_block:
            row_att_args['num_input_features'] = self.num_input_features

        att_args = cycle([row_att_args, col_att_args])
        AttentionBlocks = [row_attention_block, self.att_block]
        AttentionBlocks = cycle(AttentionBlocks)
        permute_idxs = [1, 0, 2]

        # Permute to (D, N, E)
        enc = [Permute(permute_idxs)]

        # # Attend between instances N
        # enc.append(next(AttentionBlocks)(
        #     self.dim_input, self.dim_hidden, self.dim_hidden,
        #     **next(att_args)))

        # Start with attention over the instances

        # for i in range(1, self.stacking_depth):
        for i in range(self.stacking_depth):
            # alternatingly attend between features D and instances N
            enc.append(next(AttentionBlocks)(
                self.dim_hidden, self.dim_hidden, self.dim_hidden,
                **next(att_args)))
            enc.append(Permute(permute_idxs))

        # # finally, project back from the hidden dimension to the input
        # # dimension
        # enc.append(Permute(permute_idxs))
        # enc.append(next(AttentionBlocks)(
        #     self.dim_hidden, self.last_embed, self.dim_input,
        #     **next(att_args)))
        enc = nn.Sequential(*enc)

        return enc

    def get_hybrid_custom_stack_ttnn(self):
        """
        WIP - Hybrid with a custom stacking pattern.
        """
        if self.stacking_depth < 4:
            raise ValueError(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 4.')
        if self.stacking_depth % 4 != 0:
            raise ValueError('Please provide a stacking depth divisible by 4.')

        # Retrieve the specified AttentionBlock for attention over the rows
        try:
            row_attention_block = ATT_BLOCK_NAME_TO_CLASS[
                self.model_nested_row_att_block]
        except KeyError:
            raise NotImplementedError

        # Indep row attention is unsupported with the hybrid model, because
        # our row attention is performed as in flattened TTNN.
        if 'Indep' in self.model_nested_row_att_block:
            raise NotImplementedError(
                'Indep blocks cannot be used for row attention'
                ' in the Hybrid TTNN model.')

        print('Building custom hybrid transformer: col-col-col-row pattern')
        print(f'Column-wise attention block: {self.att_block.__name__}')
        print(f'Row-wise attention block: {row_attention_block.__name__}')
        rff_type = 'Gated GeLU' if self.c.model_rff_gated_gelu else 'GeLU'
        print(f'Row-wise feed-forward activation: {rff_type}')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'c': self.c}
        col_att_args = {'c': self.c}

        # Get correct number of inducing points for attention blocks.
        # There is the option to specify a different number of inducing
        # points for row and column attention.
        if getattr(self.att_block, 'has_inducing_points', False):
            col_att_args['num_inds'] = self.c.model_num_inds
        if getattr(row_attention_block, 'has_inducing_points', False):
            if (row_inds := self.c.model_num_row_inds) != -1:
                row_att_args['num_inds'] = row_inds
            else:
                row_att_args['num_inds'] = self.c.model_num_inds

        # Provide the number of columns to the column attention block when
        # additive or multiplicative encodings are enabled. These are
        # applied in the attention module, before and after the softmax
        # is applied, respectively. See ttnn_modules.py.
        if (self.c.model_att_additive_encoding or
                self.c.model_att_multiplicative_encoding):
            col_att_args['num_input_features'] = self.num_input_features

        # Perform attention over columns 3 times, then attention over rows
        att_args = cycle(
            [col_att_args, col_att_args, col_att_args, row_att_args])
        AttentionBlocks = [
            self.att_block, self.att_block, self.att_block,
            row_attention_block]
        AttentionBlocks = cycle(AttentionBlocks)

        D = self.num_input_features

        enc = []

        if self.c.model_hybrid_debug:
            enc.append(Print())

        # Already in shape (N, D, E)

        layer_index = 0

        while layer_index < self.stacking_depth:
            # Attention over the rows (every 4 blocks)
            if (layer_index + 1) % 4 == 0:
                # Reshape to flattened representation
                enc.append(ReshapeToFlat())

                if self.c.model_hybrid_debug:
                    enc.append(Print())

                # Perform attention over the rows
                enc.append(next(AttentionBlocks)(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D,
                    **next(att_args)))

                # Reshape to nested rep
                enc.append(ReshapeToNested(D=D))

                if self.c.model_hybrid_debug:
                    enc.append(Print())
            else:
                # Attention over the columns
                # Input is already in nested shape (N, D, E)
                enc.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden,
                    self.dim_hidden,
                    **next(att_args)))

                if self.c.model_hybrid_debug:
                    enc.append(Print())

            layer_index += 1

        enc = nn.Sequential(*enc)
        print(enc)
        return enc

    def get_hybrid_ttnn(self):
        """
        A hybrid, combining flattened attention over the rows and "nested"
        attention over the columns.

        This is reasonable if we don't aim to maintain column equivariance
        (which we essentially never do, because of the column-specific
        feature embeddings at the input and output of the TTNN encoder.

        This hybrid is done by concatenating the feature outputs of column
        attention and inputting them to row attention. Therefore, it requires
        reshaping between each block (which is still the case in nested
        attention)! However, it also requires splitting and concatenation.
        """
        if self.stacking_depth < 2:
            raise ValueError(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')
        if self.stacking_depth % 2 != 0:
            raise ValueError('Please provide an even stacking depth.')
        if (self.model_recurrence_interval is not None
                and self.stacking_depth % self.model_recurrence_interval != 0):
            raise ValueError(
                f'Stacking depth {self.stacking_depth} should be a multiple '
                f'of the recurrence interval '
                f'{self.model_recurrence_interval}.')

        # Retrieve the specified AttentionBlock for attention over the rows
        try:
            row_attention_block = ATT_BLOCK_NAME_TO_CLASS[
                self.model_nested_row_att_block]
        except KeyError:
            raise NotImplementedError

        # Indep row attention is unsupported with the hybrid model, because
        # our row attention is performed as in flattened TTNN.
        if 'Indep' in self.model_nested_row_att_block:
            raise NotImplementedError(
                'Indep blocks cannot be used for row attention'
                ' in the Hybrid TTNN model.')

        print('Building hybrid transformer.')
        if self.model_recurrence_interval is not None:
            print(
                f'With recurrence interval {self.model_recurrence_interval}, '
                f'shared weight setting '
                f'{self.model_recurrence_share_weights}.')

        print(f'Row-wise attention block: {row_attention_block.__name__}')
        print(f'Column-wise attention block: {self.att_block.__name__}')
        rff_type = 'Gated GeLU' if self.c.model_rff_gated_gelu else 'GeLU'
        print(f'Row-wise feed-forward activation: {rff_type}')

        # *** Construct arguments for row and column attention. ***

        row_att_args = {'c': self.c}
        col_att_args = {'c': self.c}

        # Get correct number of inducing points for attention blocks.
        # There is the option to specify a different number of inducing
        # points for row and column attention.
        if getattr(self.att_block, 'has_inducing_points', False):
            col_att_args['num_inds'] = self.c.model_num_inds
        if getattr(row_attention_block, 'has_inducing_points', False):
            if (row_inds := self.c.model_num_row_inds) != -1:
                row_att_args['num_inds'] = row_inds
            else:
                row_att_args['num_inds'] = self.c.model_num_inds

        # Provide the number of columns to the column attention block when
        # additive or multiplicative encodings are enabled. These are
        # applied in the attention module, before and after the softmax
        # is applied, respectively. See ttnn_modules.py.
        if (self.c.model_att_additive_encoding or
                self.c.model_att_multiplicative_encoding):
            col_att_args['num_input_features'] = self.num_input_features

        # Indicates that we will need to discard the initial input, X,
        # which will be carried through to the output. If we have a
        # number of recurrent blocks that do not divide evenly the total
        # number of rows, we will perform the unpacking prior to the output,
        # and will disable this flag.
        self.unpack_output = True

        # Perform attention over rows first
        att_args = cycle([row_att_args, col_att_args])
        AttentionBlocks = [row_attention_block, self.att_block]
        AttentionBlocks = cycle(AttentionBlocks)

        D = self.num_input_features

        enc = []

        if self.c.model_hybrid_debug:
            enc.append(Print())

        # Reshape to flattened representation (1, N, D*dim_input)
        enc.append(ReshapeToFlat())
        # print('dim_input * D', self.dim_input * D)
        # print('dim_hidden * D', self.dim_hidden * D)

        recurrence = self.c.model_recurrence_share_weights

        if self.c.model_type == 'hybrid-inducing' and not recurrence:
            enc = self.build_hybrid_inducing_no_weight_sharing_enc(
                enc, AttentionBlocks, att_args, D)

        elif self.c.model_type == 'hybrid-inducing' and recurrence:
            raise NotImplementedError

        elif recurrence:
            enc = self.build_hybrid_weight_sharing_enc(
                enc, AttentionBlocks, att_args, D)
        else:
            enc = self.build_hybrid_no_weight_sharing_enc(
                enc, AttentionBlocks, att_args, D)

        enc = nn.Sequential(*enc)
        return enc

    def build_hybrid_inducing_no_weight_sharing_enc(
            self, enc, AttentionBlocks, att_args, D):
        final_shape = None

        if self.c.model_hybrid_debug:
            stack = [Print()]
        else:
            stack = []

        args = dict(c=self.c)

        if self.stacking_depth == 1:
            raise NotImplementedError

        if self.c.model_use_pre_npt_rff:
            stack += [
                rFF(self.rff_dim_input, self.dim_hidden * D, self.c),
                SlotMAB(self.dim_hidden * D, self.dim_hidden * D,
                        self.dim_hidden * D, self.c,
                        self.c.model_num_inds)]
        else:
            # MAB(I, X)
            stack.append(
                SlotMAB(
                    self.dim_input, self.dim_hidden * D,
                    self.dim_hidden * D, self.c,
                    self.c.model_num_inds))

        if self.c.model_hybrid_debug:
            stack.append(Print())

        # Repeated SABs
        layer_index = 0

        while layer_index < (self.stacking_depth - 1):
            if layer_index % 2 == 1:
                # Input is already in nested shape (N, D, E)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden, self.dim_hidden,
                    **next(att_args)))

                # Reshape to flattened representation
                stack.append(ReshapeToFlat())
                final_shape = 'flat'

                if self.c.model_hybrid_debug:
                    stack.append(Print())
            else:
                # Input is already in flattened shape (1, N, D*E)

                # Attend between instances N
                # whenever we attend over the instances,
                # we consider dim_hidden = self.c.dim_hidden * D
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D,
                    **next(att_args)))

                # Reshape to nested representation
                stack.append(ReshapeToNested(D=D))
                final_shape = 'nested'

                if self.c.model_hybrid_debug:
                    stack.append(Print())

            if self.model_recurrence_interval is not None:
                # Only perform recurrence on the interval, and
                # do not apply it directly before the output
                if (((layer_index + 1) % self.model_recurrence_interval == 0)
                        and layer_index != self.stacking_depth - 1):
                    # Recurrent cross-attention block over the columns
                    if layer_index % 2 == 0:
                        if (self.c.model_att_additive_encoding or
                                self.c.model_att_multiplicative_encoding):
                            dim_att = self.num_input_features
                        else:
                            dim_att = None

                        enc.append(RecurrentBlock(
                            c=self.c, core_blocks=nn.Sequential(*stack),
                            dim_Q=self.dim_hidden,
                            dim_KV=self.dim_hidden,
                            dim_emb=self.dim_hidden,
                            dim_out=self.dim_hidden, dim_att=dim_att))
                    else:
                        enc.append(RecurrentBlock(
                            c=self.c, core_blocks=nn.Sequential(*stack),
                            dim_Q=self.dim_hidden * D,
                            dim_KV=self.dim_hidden * D,
                            dim_emb=self.dim_hidden * D,
                            dim_out=self.dim_hidden * D, dim_att=None))

                    stack = []

                # At the last layer, append the remaining layers in the
                # stack to the encoder
                elif layer_index == self.stacking_depth - 1:
                    # Deal with the output from recurrent blocks, which will
                    # pass on an unneeded instance of the initial input
                    if len(stack) > 0:
                        stack[0] = RecurrentHandler(stack[0])
                        self.unpack_output = False

                    enc += stack
                    stack = []

            # Conglomerate the stack into the encoder thus far
            else:
                enc += stack
                stack = []

            layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if final_shape != 'flat':
            enc.append(ReshapeToFlat())

        if self.c.model_use_pre_npt_rff:
            self.final_mab = MAB(
                self.rff_dim_input, self.dim_hidden, self.dim_hidden,
                self.dim_hidden, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)
        else:
            self.final_mab = MAB(
                self.dim_input, self.dim_hidden * D, self.dim_hidden * D,
                self.dim_hidden * D, self.c, dim_att=None,
                force_ablate_rff=self.c.model_ablate_final_layer_rff)

        self.final_linear = nn.Linear(
            self.dim_hidden,
            self.dim_input // self.num_input_features
            )
        self.final_flatten = ReshapeToFlat()
        self.final_unflatten = ReshapeToNested(D=D)

        return enc

    def build_hybrid_weight_sharing_enc(
            self, enc, AttentionBlocks, att_args, D):
        if self.c.model_hybrid_debug:
            stack = [Print()]
        else:
            stack = []

        layer_index = 0

        while layer_index < self.model_recurrence_interval:
            if layer_index % 2 == 1:
                # Input is already in nested shape (N, D, E)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden, self.dim_hidden,
                    **next(att_args)))

                # Reshape to flattened representation
                stack.append(ReshapeToFlat())

                if self.c.model_hybrid_debug:
                    stack.append(Print())
            else:
                # Input is already in flattened shape (1, N, D*E)

                # Attend between instances N
                # whenever we attend over the instances,
                # we consider dim_hidden = self.c.dim_hidden * D
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D,
                    **next(att_args)))

                # Reshape to nested representation
                stack.append(ReshapeToNested(D=D))

                if self.c.model_hybrid_debug:
                    stack.append(Print())

            layer_index += 1

        enc.append(RecurrentBlock(
            c=self.c, core_blocks=nn.Sequential(*stack),
            dim_Q=self.dim_hidden * D,
            dim_KV=self.dim_hidden * D,
            dim_emb=self.dim_hidden * D,
            dim_out=self.dim_hidden * D, dim_att=None,
            num_timesteps=int(
                self.stacking_depth // self.model_recurrence_interval)))

        enc.append(RecurrentHandler(ReshapeToNested(D=D)))
        self.unpack_output = False
        return enc

    def build_hybrid_no_weight_sharing_enc(
            self, enc, AttentionBlocks, att_args, D):
        final_shape = None

        if self.c.model_hybrid_debug:
            stack = [Print()]
        else:
            stack = []

        layer_index = 0

        while layer_index < self.stacking_depth:
            if layer_index % 2 == 1:
                # Input is already in nested shape (N, D, E)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden, self.dim_hidden,
                    **next(att_args)))

                # Reshape to flattened representation
                stack.append(ReshapeToFlat())
                final_shape = 'flat'

                if self.c.model_hybrid_debug:
                    stack.append(Print())
            else:
                # Input is already in flattened shape (1, N, D*E)

                # Attend between instances N
                # whenever we attend over the instances,
                # we consider dim_hidden = self.c.dim_hidden * D
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D,
                    **next(att_args)))

                # Reshape to nested representation
                stack.append(ReshapeToNested(D=D))
                final_shape = 'nested'

                if self.c.model_hybrid_debug:
                    stack.append(Print())

            if self.model_recurrence_interval is not None:
                # Only perform recurrence on the interval, and
                # do not apply it directly before the output
                if (((layer_index + 1) % self.model_recurrence_interval == 0)
                        and layer_index != self.stacking_depth - 1):
                    # Recurrent cross-attention block over the columns
                    if layer_index % 2 == 0:
                        if (self.c.model_att_additive_encoding or
                                self.c.model_att_multiplicative_encoding):
                            dim_att = self.num_input_features
                        else:
                            dim_att = None

                        enc.append(RecurrentBlock(
                            c=self.c, core_blocks=nn.Sequential(*stack),
                            dim_Q=self.dim_hidden,
                            dim_KV=self.dim_hidden,
                            dim_emb=self.dim_hidden,
                            dim_out=self.dim_hidden, dim_att=dim_att))
                    else:
                        enc.append(RecurrentBlock(
                            c=self.c, core_blocks=nn.Sequential(*stack),
                            dim_Q=self.dim_hidden * D,
                            dim_KV=self.dim_hidden * D,
                            dim_emb=self.dim_hidden * D,
                            dim_out=self.dim_hidden * D, dim_att=None))

                    stack = []

                # At the last layer, append the remaining layers in the
                # stack to the encoder
                elif layer_index == self.stacking_depth - 1:
                    # Deal with the output from recurrent blocks, which will
                    # pass on an unneeded instance of the initial input
                    if len(stack) > 0:
                        stack[0] = RecurrentHandler(stack[0])
                        self.unpack_output = False

                    enc += stack
                    stack = []

            # Conglomerate the stack into the encoder thus far
            else:
                enc += stack
                stack = []

            layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if final_shape == 'flat':
            enc.append(ReshapeToNested(D=D))

        return enc

    def get_column_only_ttnn(self):
        """Ablation version. Like Nested TTNN but only column attention."""

        if self.stacking_depth < 2:
            raise Exception(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')

        # Retrieve the specified AttentionBlock for attention over the rows
        try:
            row_attention_block = ATT_BLOCK_NAME_TO_CLASS[
                self.model_nested_row_att_block]
        except KeyError:
            raise NotImplementedError

        print('Building nested transformer. DEBUG: Only Col Attention.')
        print(f'Row-wise attention block: {self.att_block.__name__}')
        print(f'Column-wise attention block: {self.att_block.__name__}')

        # *** Construct arguments for row and column attention. ***

        col_att_args = {'c': self.c}

        # Get correct number of inducing points for attention blocks.
        # There is the option to specify a different number of inducing
        # points for row and column attention.
        if getattr(self.att_block, 'has_inducing_points', False):
            col_att_args['num_inds'] = self.c.model_num_inds

        # Provide the number of columns to the column attention block when
        # additive or multiplicative encodings are enabled. These are
        # applied in the attention module, before and after the softmax
        # is applied, respectively. See ttnn_modules.py.
        if (self.c.model_att_additive_encoding or
                self.c.model_att_multiplicative_encoding):
            col_att_args['num_input_features'] = self.num_input_features

        # Provide number of columns to Indep attention block, which is used
        # for attention over the rows (i.e. an indepedent attention map is
        # learned for each column).

        att_args = cycle([col_att_args])
        AttentionBlocks = [self.att_block]
        AttentionBlocks = cycle(AttentionBlocks)

        # Attend between instances N
        enc = [next(AttentionBlocks)(
            self.dim_hidden, self.dim_hidden, self.dim_hidden,
            **next(att_args))]

        depth = self.stacking_depth
        if self.c.model_ablate_col_attention_only_no_replace:
            depth = depth // 2

        for i in range(1, depth):
            # alternatingly attend between features D and instances N
            enc.append(next(AttentionBlocks)(
                self.dim_hidden, self.dim_hidden, self.dim_hidden,
                **next(att_args)))

        # finally, project back from the hidden dimension to the input
        # dimension
        # enc.append(next(AttentionBlocks)(
        #     self.dim_hidden, self.last_embed, self.dim_input,
        #     **next(att_args)))

        enc = nn.Sequential(*enc)

        return enc

    def get_hierarchical_i_npt(self):
        '''Gets hierarchical perceiver as implemented in the paper
        https://arxiv.org/abs/2202.10890.
        NOTE: this model does not use the standard parameters of stacking
        depth etc., and instead realies on parameters specified at the
        start of the hpc_modules.py file
        '''
        if self.c.model_type == 'h-npt-large':
            self.dim_hidden = 128

        self.final_linear = nn.Linear(self.dim_hidden, self.dim_input)
        enc = Hierarchical_Perceiver(self.dim_input, self.c)

        return enc

    # @profile
    def forward(self, X_ragged, X_labels=None, eval_model=None):
        """Provide X_labels only for supervised contrastive regularization."""
        if self.image_patcher is not None:
            X = self.image_patcher.encode(X_ragged)
            in_dims = [X.size(0), X.size(1), -1]
        else:
            in_dims = [X_ragged[0].shape[0], len(X_ragged), -1]

            if self.c.model_use_pre_npt_rff:
                # Don't use the in_embedding
                X = X_ragged
                X = torch.cat(X, 1)

            elif self.time_code:
                X = [self.in_embedding[t](X_ragged[t])
                     for t in self.time_code[0]]
                for time_subset in self.time_code[1:]:
                    X += [self.in_embedding[self.time_code[1][i]](X_ragged[t])
                          for i, t in enumerate(time_subset)]
                X = torch.stack(X, 1)

            else:
                # encode ragged input array D x {(NxH_j)}_j to NxDxE)
                X = [embed(X_ragged[i])
                     for i, embed in enumerate(self.in_embedding)]
                X = torch.stack(X, 1)

        # Compute feature type (cat vs numerical) embeddings, and add them
        if self.feature_type_embedding is not None:
            feature_type_embeddings = self.feature_type_embedding(
                self.feature_types)

            # Add a batch dimension (the rows)
            feature_type_embeddings = torch.unsqueeze(
                feature_type_embeddings, 0)

            # Tile over the rows
            feature_type_embeddings = feature_type_embeddings.repeat(
                X.size(0), 1, 1)

            # Add to X
            X = X + feature_type_embeddings

        # Compute feature index embeddings, and add them
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices)

            # Add a batch dimension (the rows)
            feature_index_embeddings = torch.unsqueeze(
                feature_index_embeddings, 0)

            # Tile over the rows
            feature_index_embeddings = feature_index_embeddings.repeat(
                X.size(0), 1, 1)

            # Add to X
            X = X + feature_index_embeddings

        if self.prototypes is not None:
            # Add prototypes to rows.
            in_dims[0] += self.prototypes.num
            X = self.prototypes.concat(X)

        # Embedding tensor currently has shape (N x D x E)

        # Follow BERT in applying LayerNorm -> Dropout on embeddings
        if self.c.model_use_pre_npt_rff:
            if self.in_embedding_layer_norm is not None:
                X = self.in_embedding_layer_norm(X)
        else:
            if self.embedding_layer_norm is not None:
                X = self.embedding_layer_norm(X)

        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        # 0 -> no permutation
        # 1 -> fixed permutation
        # 2 -> rand permutation
        if self.c.image_test_patch_permutation != 0:
            num_cols = X.size(1)
            if self.c.image_test_patch_permutation == 1:
                row_permutation = np.random.permutation(np.arange(num_cols))
                inverse_row_permutation = np.argsort(row_permutation)

            # For each row of the input, permute with above along the D axis
            new_X = []

            if self.c.image_test_patch_permutation == 2:
                row_inverse_perms = []
            for n in range(X.size(0)):  # Rows
                if self.c.image_test_patch_permutation == 1:
                    new_X.append(X[n, row_permutation, :])

                if self.c.image_test_patch_permutation == 2:
                    rand_perm = np.random.permutation(np.arange(num_cols))
                    row_inverse_perms.append(np.argsort(rand_perm))
                    new_X.append(X[n, rand_perm, :])

            X = torch.stack(new_X)

        if self.is_flattened:
            # flatten to shape Nx(D*E)
            X = torch.flatten(X, start_dim=1, end_dim=-1)
            # add batch dimension at dim 0, attention will be performed
            # between the entries of dim 1. We need this batch dimension b/c
            # of how the multiple attention heads are implemented (to remain
            # compatible with the nested transformer, which batches over the
            # 0th dim)
            X = X.unsqueeze(0)

            if True:
                a = 1

        # apply TTNN
        if self.c.model_type == 'i-npt':
            X_new = self.enc(X)
            X = self.final_mab(X, X_new) #TODO: we need to make sure the out layer does not involve an MLP
            X = self.final_linear(X) #TODO: stop final output 

        elif self.c.model_type == 'iterative-inducing':
            X_data = self.enc(X)
            X_attr = self.attribute_enc(X_data)
            X_data = self.recover_attr_MAB(X_data, X_attr)
            X_data = self.final_flatten(X_data)
            X = self.recover_batch_MAB(X, X_data)
            X = self.final_linear(X)

        elif self.c.model_type == 'hybrid-inducing':
            X_new = self.enc(X)
            X = self.final_flatten(X)
            X = self.final_mab(X, X_new)
            X = self.final_unflatten(X)
            X = self.final_linear(X)

        elif self.c.model_type == 'h-npt-large':
            X = self.enc(X)
            X = self.final_linear(X)

        else:
            X = self.enc(X)

        if (self.model_recurrence_interval is not None
                and self.model_type == 'hybrid'
                and self.unpack_output):
            # Ignore the initial X, which has been carried through
            # the model for recurrence.
            _, X = X

        # if self.is_nested and self.stacking_depth % 2 != 0:
        if self.has_column_attention and (X.shape[1] == in_dims[0]):
            # for uneven stacking_depth, need to permute one last time
            # to obtain output of shape (N, D, E)
            X = X.permute([1, 0, 2])
        elif not self.has_column_attention:
            # reshape (1, N, dim_out) to (N, D, -1)
            X = X.reshape(in_dims)

        if self.prototypes is not None:
            # Discard prototypes
            X = self.prototypes.discard(X)

        # Dropout before final projection (follows BERT, which performs
        # dropout before e.g. projecting to logits for sentence classification)
        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        # Permute back to the original order of patches
        # 0 -> no permutation
        # 1 -> fixed permutation
        # 2 -> rand permutation
        if self.c.image_test_patch_permutation != 0:
            # For each row of the input, permute indep along the D axis
            new_X = []
            for i, n in enumerate(range(X.size(0))):  # Rows
                if self.c.image_test_patch_permutation == 1:
                    new_X.append(X[n, inverse_row_permutation, :])
                elif self.c.image_test_patch_permutation == 2:
                    new_X.append(X[n, row_inverse_perms[i], :])
            X = torch.stack(new_X)

        if self.image_patcher is None:
            # project back to ragged (dimensions D x {(NxH_j)}_j )
            # Is already split up across D
            if self.time_code:
                X_ragged = [self.out_embedding[t](
                    X[:, t]) for t in self.time_code[0]]
                for time_subset in self.time_code[1:]:
                    X_ragged += [self.out_embedding[self.time_code[1][i]](
                        X[:, t]) for i, t in enumerate(time_subset)]

            else:
                X_ragged = [de_embed(X[:, i]) for i, de_embed in enumerate(
                    self.out_embedding)]
        else:
            X_ragged = self.image_patcher.decode(X)

        if self.c.model_supcon_regularizer and X_labels is not None:
            # Assumes we have two data augmentation views
            # f1, f2 = torch.split(X, [X.size(0), X.size(0)], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            supcon_loss = self.supcon_regularizer(
                X[:, self.supcon_target_col].unsqueeze(1), X_labels)
            return X_ragged, supcon_loss

        return X_ragged


class Permute(nn.Module):
    """Permutation as nn.Module to include in nn.Sequential."""
    def __init__(self, idxs):
        super(Permute, self).__init__()
        self.idxs = idxs

    def forward(self, X):
        return X.permute(self.idxs)


class ReshapeToFlat(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (1, N, D*E)."""
    def __init__(self):
        super(ReshapeToFlat, self).__init__()

    @staticmethod
    def forward(X):
        return X.reshape(1, X.size(0), -1)


class ReshapeToNested(nn.Module):
    """Reshapes a tensor of shape (1, N, D*E) to (N, D, E)."""
    def __init__(self, D):
        super(ReshapeToNested, self).__init__()
        self.D = D

    def forward(self, X):
        return X.reshape(X.size(1), self.D, -1)


class RecurrentBlock(nn.Module):
    def __init__(self, c, core_blocks: nn.Sequential,
                 dim_Q: int, dim_KV: int, dim_emb: int, dim_out: int,
                 dim_att: Optional[int] = None, num_timesteps: int = 1):
        """
        Compute Perceiver-style recurrent cross-attention.

        This function takes an input X, and computes:
            HRB(X) = MAB(core_blocks(X), X)

        Below, N = # rows, D = # columns, E = dim_feature_embedding.
        core_blocks(X) has shape (*, d_1, d_2). Note that this covers
            (1) the flattened row cross-attention case:
                core_blocks(X).shape = (1, N, D*E)
            (2) the nested row cross-attention case:
                core_blocks(X).shape = (D, N, E)
            (3) the column cross-attention case:
                core_blocks(X).shape = (N, D, E)

        Args:
            core_blocks: List[nn.Module], a sequence of Attention Blocks and
                transformations such as Permute(), ReshapeToNested(), or
                ReshapeToFlat(). These form the backbone of the RecurrentTTNN
                model.
            dim_Q: int, the last dimension of core_blocks(X) at input to cross-
                attention.
            dim_KV: int, the last dimension of X at input to cross-attention.
            dim_emb: int, Q, K, and V in cross-attention are all embedded to
                this dimensionality.
            dim_out: int, the output dimensionality of the MAB. i.e., it will
                have shape (*, d_1, dim_out).
            dim_att: Optional[int], needs to be specified when we aim to learn
                and apply either additive encodings to the attention weight
                tensor (pre-softmax) or multiplicative encodings to the
                attention score tensor (post-softmax).

                NOTE: this is only valid when performing attention over the
                 columns, as in nested or hybrid TTNN
                 (else, it would break row equivariance).
            num_timesteps: int, number of recurrent timesteps, i.e., the
                number of times that the RecurrentBlock should be applied.
                If >1, we do not apply the cross-attention in the final loop.
        """
        super(RecurrentBlock, self).__init__()
        self.core_blocks = core_blocks
        self.mab = MAB(
            dim_Q, dim_KV, dim_emb, dim_out, c, dim_att=dim_att)

        assert num_timesteps >= 1
        self.num_timesteps = num_timesteps

    def forward(self, inp):
        if isinstance(inp, torch.Tensor):
            # This RecurrentBlock is the first in the full network,
            # and has only been supplied the initial input X
            X = H = inp
        elif isinstance(inp, tuple):
            # This RecurrentBlock is carrying through the
            # input X and intermediate state H
            X, H = inp
        else:
            raise NotImplementedError

        timestep = 0

        while timestep < self.num_timesteps:
            # Apply the backbone to the intermediate state H (which in the
            # case of the first recurrent block, is just the input)
            core_blocks_output = self.core_blocks(H)

            # May have to apply extra transformation to initial input
            if core_blocks_output.size(0) == 1 and X.size(0) != 1:
                # Reshape input to flattened dimensions (1, N, D*E)
                X = ReshapeToFlat()(X)
            elif core_blocks_output.size(1) != X.size(1):
                # Reshape input to nested dimensions (N, D, E)
                X = ReshapeToNested(D=core_blocks_output.size(1))(X)

            # If we only apply this block once, we are not sharing
            # weights and the application of the final cross-attention
            # is already avoided.
            # Otherwise (in the shared weight setting), we apply the
            # cross-attention at all timesteps except the last.
            if self.num_timesteps == 1 or timestep < self.num_timesteps - 1:
                # Apply cross-attention between the newly-computed intermediate
                # state, and the initial input X
                recurrent_output = self.mab(core_blocks_output, X)

            timestep += 1

        # Return the initial input, and recurrent output for the
        # next recurrent block.
        return X, recurrent_output


class RecurrentHandler(nn.Module):
    def __init__(self, block):
        super(RecurrentHandler, self).__init__()
        self.block = block

    def forward(self, inp):
        if isinstance(inp, torch.Tensor):
            H = inp
        elif isinstance(inp, tuple):
            _, H = inp
        else:
            raise NotImplementedError

        return self.block(H)


class Print(nn.Module):
    def __init__(self, flag='Debug'):
        super(Print, self).__init__()
        self.flag = flag

    def forward(self, x):
        print(self.flag, x.shape)
        return x
