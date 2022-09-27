from itertools import cycle

import torch
import torch.nn as nn

from ttnn.model.ttnn_modules import SlotMAB
from ttnn.utils.encode_utils import torch_cast_to_dtype


class ImagePatcher(nn.Module):
    def __init__(
            self, dim_hidden, input_feature_dims, c, patcher_type):
        super(ImagePatcher, self).__init__()

        self.c = c

        # Global setting, avoids non-patching logic in TTNN init
        self.model_image_patching = True

        D = len(input_feature_dims)  # Flattened size of an image

        # We reduce what the core model sees to a sequence of patches
        # (unordered, but with patch index embeddings), until the decoder
        self.image_n_patches = self.c.model_image_n_patches

        # Includes the target column
        self.num_input_features = self.image_n_patches + 1

        # Options: {'linear', 'conv', 'slot-mab'}
        self.image_patch_type = self.c.model_image_patch_type

        # Share embedding weights across patches, or separate
        self.image_share_embed = self.c.model_image_share_embed

        # e.g. 3 for RGB
        self.image_n_channels = self.c.model_image_n_channels

        # If we use BERT augmentation, this must be 2, for
        # the continuous pixel intensity and the mask value.
        # Otherwise, this should be 1.
        self.dim_intensity = 1 + bool(self.c.model_bert_augmentation)
        self.dim_target_col = self.c.model_image_n_classes + bool(
            self.c.model_bert_augmentation)

        # Exclude target column (we assume it is right concatenated)
        image_input_shape = (D - 1, self.dim_intensity)

        # The number of patches must divide the number of pixels
        assert image_input_shape[0] % self.image_n_patches == 0

        # This is in raw intensities, i.e. counting each pixel in an
        # RGB image thrice
        self.patch_size = image_input_shape[0] // self.image_n_patches

        # Compute resizing constants
        n_features = len(input_feature_dims) - 1
        assert n_features % self.image_n_channels == 0

        if patcher_type == 'conv' or patcher_type == 'linear':
            # H = height, note that we are expecting square images for now
            # H = height = W = width
            flattened_image_size = n_features // self.image_n_channels
            self.image_H = int(flattened_image_size ** 0.5)
            assert flattened_image_size // self.image_H == self.image_H

            # Get number of rows of patches
            n_patches_per_side = self.image_n_patches ** 0.5
            assert int(n_patches_per_side) == n_patches_per_side
            n_patches_per_side = int(n_patches_per_side)

            # Get length of patches
            # (i.e. each patch is patch_side_length x patch_side_length)
            assert self.image_H % n_patches_per_side == 0
            self.patch_side_length = self.image_H // n_patches_per_side

        # ### Embeddings ###

        # Always use a linear out-embedding
        if self.image_share_embed:
            # Output into the number of intensities in a patch
            # (no mask dim needed), applied in a sliding fashion
            self.out_feature_embedding = nn.ModuleList([
                nn.Linear(dim_hidden, self.patch_size)])
        else:
            # Separate linear embedding for each patch
            self.out_feature_embedding = nn.ModuleList([
                nn.Linear(dim_hidden, self.patch_size)
                for _ in range(self.image_n_patches)])

        self.out_target_embedding = nn.Linear(
            dim_hidden, c.model_image_n_classes)

    def decode(self, X):
        # We receive a tensor of shape (N, n_patches + 1, E)

        # Feature Patch De-Embedding
        if self.image_share_embed:
            de_embeds = cycle(self.out_feature_embedding)
        else:
            de_embeds = self.out_feature_embedding

        X_ragged = []

        # Projects each batched feature patch of shape (N, E) to (N,
        for patch_index in range(X.shape[1] - 1):
            # X_patch.shape = (N, E)
            X_patch = X[:, patch_index, :]

            # de_embed.shape = (E, p) where p = patch size
            de_embed = next(de_embeds)

            # X_de_embed.shape = (N, p)
            X_de_embed = de_embed(X_patch)

            # Split into p columns of shape (N, 1)
            X_de_embed = torch.split(X_de_embed, 1, dim=1)
            X_ragged += X_de_embed

        # Append projection of target column
        X_ragged.append(self.out_target_embedding(X[:, -1, :]))

        return X_ragged

    def get_ttnn_attrs(self):
        """Send a few key attributes back to the main model."""
        return {'num_input_features': self.num_input_features,
                'image_n_patches': self.image_n_patches,
                'patch_size': self.patch_size}

    def preprocess_flattened_image(self, X_ragged):
        """
        Prior to applying the Linear and Conv transforms, we wish to reshape
        our features, which constitute the image:
            * D = total number of columns (including the target)
            (N, D - 1, dim_intensity)
            where dim_intensity is 2 if we are using masking, 1 otherwise
            to (N, (D - 1) // n_channels, dim_intensity * n_channels)

        This is necessary because, e.g., CIFAR-10 flattens images to be of
            format 1024 R, 1024 G, 1024 B. We must reshape to make sure
            the patching has the correct receptive fields.

        Returns:
            Reshaped X_features, X_target column
        """
        # Shape (N, D - 1, dim_intensity)
        # where dim_intensity = 2 if we have continuous pixel intensity + mask
        # or 1 if we just have the pixel intensity (no BERT augmentation mask)
        X_features = torch.stack(X_ragged[:-1], 1)

        # Reshape to (N, (D - 1) // n_channels, dim_intensity * n_channels)
        X_features = torch.reshape(
            X_features,
            (X_features.size(0),
             X_features.size(1) // self.image_n_channels,
             self.dim_intensity * self.image_n_channels))

        # Shape (N, 1, H_j) where H_j = num_categories + bool(is_mask)
        # (e.g. 2, for image regression with BERT augmentation)
        X_target = X_ragged[-1]

        return X_features, X_target


class LinearImagePatcher(ImagePatcher):
    def __init__(self, input_feature_dims, dim_feature_embedding, dim_hidden, c):
        super(LinearImagePatcher, self).__init__(
            dim_hidden, input_feature_dims, c, patcher_type='linear')

        self.patch_n_pixels = self.patch_side_length * self.patch_side_length
        pixel_input_dims = self.dim_intensity * self.image_n_channels

        # Each patch embedding should be shape
        # (patch_n_pixels, (1 + bool(is_mask)) * n_channels, dim_feature_embedding)
        if self.image_share_embed:
            self.in_feature_embedding = nn.ParameterList([
                nn.Parameter(torch.empty(
                    self.patch_n_pixels, pixel_input_dims,
                    dim_feature_embedding))])
        else:
            self.in_feature_embedding = nn.ParameterList([
                nn.Parameter(torch.empty(
                    self.patch_n_pixels, pixel_input_dims,
                    dim_feature_embedding))
                for _ in range(self.image_n_patches)])

        for embed in self.in_feature_embedding:
            nn.init.xavier_uniform_(embed)

        self.in_target_embedding = nn.Linear(
            self.dim_target_col, dim_feature_embedding)

    def encode(self, X_ragged):
        # Feature Patch Embedding
        # Embed to a list of n_patch tensors,
        # each of size (N, dim_feature_embedding)

        X_features, X_target = self.preprocess_flattened_image(X_ragged)

        if self.image_share_embed:
            embeds = cycle(self.in_feature_embedding)
        else:
            embeds = self.in_feature_embedding

        X_embeds = []
        for pixel_index in range(0, X_features.shape[1], self.patch_n_pixels):
            # Projection:
            # n: batch dimension, number of rows
            # p: patch size in number of locations (e.g., num RGB pixels)
            # h: dim_intensity * n_channels
            #       = (1 + 1) * n_channels if we use BERT masking,
            #       = 1 * n_channels otherwise
            # e: dim_feature_embedding, TTNN hidden dimensions

            # X_input.shape = (n, p, h)
            X_input = X_features[
                :, pixel_index:pixel_index+self.patch_n_pixels, :]

            # embed.shape = (p, h, e)
            embed = next(embeds)

            X_embeds.append(torch.einsum('nph,phe->ne', X_input, embed))

        X_embeds.append(self.in_target_embedding(X_target))
        X_embed = torch.stack(X_embeds, 1)

        return X_embed


class ConvImagePatcher(ImagePatcher):
    def __init__(self, input_feature_dims, dim_feature_embedding, dim_hidden, c):
        super(ConvImagePatcher, self).__init__(
            dim_hidden, input_feature_dims, c, patcher_type='conv')
        # TODO (possible): untie convs across image (but why not just use
        #   untied linear embeddings?)

        raise NotImplementedError

        # Conv applied to full image
        # Similar to first ResNet50 conv, but stride=(1,1) instead of (2,2)
        self.in_feature_conv = nn.Sequential(
            nn.Conv2d(
                self.image_n_channels + 1, dim_feature_embedding,
                kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
            nn.BatchNorm2d(dim_feature_embedding),
            nn.GELU())

        # Used to combine the patch_len x patch_len embeddings, each of size
        # dim_feature_embedding, into one embedding for the patch
        self.in_feature_linear = nn.Parameter(
            torch.empty(self.patch_side_length, self.patch_side_length))
        nn.init.xavier_uniform_(self.in_feature_linear)

        self.in_target_embedding = nn.Linear(
            self.c.model_image_n_classes + 1, dim_feature_embedding)

    def encode(self, X_ragged):
        X_features, X_target = self.preprocess_flattened_image(X_ragged)

        # Reshape to the standard image task shape (with masks included)
        # (N, 2 * n_channels, H, W) (assume square images for now)
        X_features = torch.reshape(
            X_features, (X_features.size(0), 2 * self.image_n_channels,
                         self.image_H, self.image_H))

        # Apply ResNet50-like Conv, BatchNorm, GeLU
        X_features = self.in_feature_conv(X_features)

        # Now have shape (N, E, H, W) where E = dim_feature_embedding
        # Collect linear combination of embeddings per-patch
        X_embeds = []

        for i in range(0, X_features.size(2), self.patch_side_length):
            for j in range(0, X_features.size(3), self.patch_side_length):
                # Linear Combination:
                # n: batch dimension, number of rows
                # k: patch side length
                # h: 2, continuous pixel intensity + mask dimension
                # e: dim_feature_embedding, TTNN hidden dimensions

                # X_input.shape = (n, e, k, k)
                X_patch = X_features[
                          :, :, i: i +self.patch_side_length,
                          j: j +self.patch_side_length]

                # embed.shape = (k, k)
                embed = self.in_feature_linear

                X_embeds.append(torch.einsum('nekk,kk->ne', X_patch, embed))

        # Shape (N, 1, H_j) where H_j = num_categories + 1
        # (or 2, for image regression)
        X_target = X_ragged[-1]
        X_embeds.append(self.in_target_embedding(X_target))
        X_embed = torch.stack(X_embeds, 1)

        return X_embed


class SlotMABImagePatcher(ImagePatcher):
    def __init__(self, input_feature_dims, dim_feature_embedding, dim_hidden, c, device):
        super(SlotMABImagePatcher, self).__init__(
            dim_hidden, input_feature_dims, c, patcher_type='slot-mab')

        self.in_feature_linear = nn.Linear(
            self.dim_intensity, dim_feature_embedding)
        self.intensity_indices = torch_cast_to_dtype(
            torch.arange(len(input_feature_dims), device=device), 'long')
        self.intensity_index_embedding = nn.Embedding(
            len(input_feature_dims), dim_feature_embedding)

        # TODO: if we do stochastic label masking, may want to throw the
        #   targets into the SlotMAB --
        #   else they should all get the same embedding across the rows
        self.slot_mab = SlotMAB(
            dim_in=dim_feature_embedding, dim_emb=dim_feature_embedding,
            dim_out=dim_feature_embedding, c=c, num_inds=self.image_n_patches,
            num_input_features=len(input_feature_dims) - 1)

        self.in_target_embedding = nn.Linear(
            self.dim_target_col, dim_feature_embedding)

    def encode(self, X_ragged):
        # Shape (N, D - 1, 2) where 2 is for continuous pixel intensity + mask
        X_features = torch.stack(X_ragged[:-1], 1)

        # Project all feature columns to dim_feature_embedding space
        X_features = self.in_feature_linear(X_features)

        # Add per-intensity learned positional embedding
        # e.g. for CIFAR-10 we have 32 x 32 x 3 = 3072 intensities and
        # unique embeddings (+ 1 for the target column)
        intensity_index_embedding = self.intensity_index_embedding(
            self.intensity_indices[:-1])

        # Add a batch dimension (the rows)
        intensity_index_embedding = torch.unsqueeze(
            intensity_index_embedding, 0)

        # Tile over the rows
        intensity_index_embedding = intensity_index_embedding.repeat(
            X_features.size(0), 1, 1)

        # Add to X
        X_features = X_features + intensity_index_embedding

        # Perform SlotMAB over X -> shape (N, n_inducing_points=n_patches, E)
        X_features = self.slot_mab(X_features)

        # Shape (N, 1, H_j) where H_j = num_categories + 1
        # (or 2, for image regression)
        X_target = X_ragged[-1]
        X_target = self.in_target_embedding(X_target)
        X_target = torch.unsqueeze(X_target, 1)

        # Add target index embedding
        target_index_embedding = self.intensity_index_embedding(
            self.intensity_indices[-1])

        # Add a batch dimension (the rows)
        target_index_embedding = torch.unsqueeze(
            target_index_embedding, 0)

        # Tile over the rows
        target_index_embedding = target_index_embedding.repeat(
            X_features.size(0), 1, 1)

        X_target = X_target + target_index_embedding
        X_embed = torch.cat((X_features, X_target), dim=1)

        return X_embed
