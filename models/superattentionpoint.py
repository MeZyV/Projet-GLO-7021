import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    :param x: torch.Tensor representing the image of shape [B, C, H, W]
    :param patch_size: Number of pixels per dimension of the patches (integer)
    :param flatten_channels: If True, the patches will be returned in a flattened format as a feature vector instead of a image grid.
    :return: return x's patches
    """
    # B, C, H, W = x.shape
    # x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    # x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    # x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    # if flatten_channels:
    #     x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    x = F.pixel_unshuffle(x, patch_size)  # [B, H_c * W_c, p_H, p_W]
    x = x.flatten(2, 3)  # [B, H_c * W_c, p_H * p_W]
    x = x.transpose(1, 2)  # [B, p_H * p_W, H_c * W_c]

    return x


def sequence_to_img(x, batch_size, patched_img_size, embed_dim, flatten_channels=True):
    """
    :param x: torch.Tensor representing the sequence of patches [B, H_c * W_c, P**2]
    :param batch_size: Batch size
    :param patched_img_size: Number of patches per dimension of the image (tuple)
    :param embed_dim: Dimension of the embedding
    :param flatten_channels: If True, the patches will be returned in a flattened format as a feature vector instead of a image grid.
    :return: return x's patches
    """
    x = x.reshape((batch_size, patched_img_size[0], patched_img_size[1], embed_dim))
    x = x.permute(0, 3, 1, 2)
    return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        :param embed_dim: Dimensionality of input and attention feature vectors
        :param hidden_dim: Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)
        :param num_heads: Number of heads to use in the Multi-Head Attention block
        :param dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class SuperAttentionPointNet(torch.nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_heads,
            patch_size,
            img_size,
            desc=False,
            dropout=0.0,
    ):
        """
        :param embed_dim: Dimensionality of the input feature vectors to the Transformer
        :param hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
        :param num_channels: Number of channels of the input (3 for RGB)
        :param num_heads: Number of heads to use in the Multi-Head Attention block per layers
        :param patch_size: Number of pixels that the patches have per dimension
        :param img_size: Size of an image for -> Maximum number of patches on the height an image can have
        :param dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.patched_img_size = ((img_size[0] // patch_size), (img_size[1] // patch_size))
        num_patches = self.patched_img_size[0] * self.patched_img_size[1]

        self.embed_dim = embed_dim

        # Layers/Networks
        self.input_layer = nn.Linear(patch_size ** 2, embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_h, dropout=dropout) for num_h in num_heads)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Detector Head.
        self.detector = nn.Sequential(OrderedDict([
            ('output_layer', nn.Linear(embed_dim, 65)),
            ('output_fn', nn.ReLU(inplace=True))
        ]))

        # Descriptor Head.
        self.superpoint_bool = desc

    def forward(self, x, dense=True):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)

        B, _, _ = x.shape
        x = self.input_layer(x)

        # Add positional encoding
        x = x + self.pos_embedding

        # Apply Transformer
        x = self.dropout(x)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)

        # Sequence to patched image
        x = sequence_to_img(x, B, self.patched_img_size, self.embed_dim)

        # Detector Head.
        semi = self.detector(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # if we want a heatmap the size of the input image.
        if not dense:
            # Softmax.
            # "channel-wise Softmax" non-learned transformation
            # Not used to compute loss
            dense = F.softmax(semi, 1)
            # Remove dustbin.
            nodust = dense[:, :-1, :, :]
            # Upsampling
            semi = F.pixel_shuffle(nodust, self.patch_size)

        # Descriptor Head.
        # if we want superpoint model:
        desc = None
        if self.superpoint_bool:
            desc = x
            # if we want a descriptor the size of the input image.
            if not dense:
                n, c, h, w = semi.size()
                desc = F.interpolate(desc, size=(h, w), mode='bicubic')
                dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
                desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return semi, desc
