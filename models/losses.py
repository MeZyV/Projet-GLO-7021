import torch
import torch.nn as nn
import kornia as K
import torch.nn.functional as F
import numpy as np


class DectectorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, semi: torch.Tensor, true_labels: torch.Tensor, v_mask: torch.Tensor = None):
        # TODO: Manage device (cause we are creating tensors inside the forward)
        """
        Forward pass compute detector loss

        :param true_labels: Ground truth interest points pytorch tensor shaped N x 1 x H x W.
        :param semi: Dense detector decoder output pytorch tensor shaped N x 65 x Hc x Wc.
        :param v_mask: valid_mask size of true_labels
        :return: Detector loss.
        """
        n, _, h, w = true_labels.size()
        _, c, h_c, w_c = semi.size()
        block_size = 8

        true_labels = true_labels.type(torch.float32)
        # True labels to dense, serves for the fully-convolutional cross-entropy loss
        convolution_labels = F.pixel_unshuffle(true_labels, block_size)
        # Channels = Classes
        convolution_labels = convolution_labels.permute(0, 2, 3, 1)
        # Add dustbin channel to labels (don't know why factor 2)
        # Add dustbin channel to labels (factor 2 to save the labels with the future noise)
        convolution_labels = torch.cat([2 * convolution_labels, torch.ones((n, h_c, w_c, 1))], dim=3)
        # If two ground truth corner positions land in the same bin
        # then we randomly select one ground truth corner location
        noise = torch.rand(convolution_labels.size()) * 0.1
        # Get labels
        labels = torch.argmax(convolution_labels + noise, dim=3)

        # Define valid mask
        if v_mask is not None:
            valid_mask = v_mask.type(torch.float32)
        else:
            valid_mask = torch.ones_like(true_labels, dtype=torch.float32)
        # Adjust valid_mask
        valid_mask = F.pixel_unshuffle(true_labels, block_size)
        valid_mask = valid_mask.permute(0, 2, 3, 1)
        valid_mask = torch.prod(valid_mask, dim=3)
        labels[valid_mask == 0] = 65

        # Get loss (ignore dustbin)
        # Get loss
        loss = nn.CrossEntropyLoss(ignore_index=65)
        output = loss(semi, labels)
        return output


class DescriptorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, DESC, warp_DESC, H, H_invert, v_mask):
        H_invert = H_invert.unsqueeze(1)
        desc = DESC.permute(0, 2, 3, 1)
        B, Hc, Wc = tuple(desc.size())[:3]

        # create grid of center of HcxWc region
        cords = torch.stack(torch.meshgrid(torch.range(0, Hc - 1), torch.range(0, Wc - 1)), dim=-1).type(
            torch.int32).to(DEVICE)
        cords = cords.unsqueeze(0)
        cords = cords * 8 + 4

        # change from ij to xy cords to warp grid
        xy_cords = torch.cat((cords[:, :, :, 1].unsqueeze(3), cords[:, :, :, 0].unsqueeze(3)), dim=-1)
        xy_warp_cords = K.geometry.warp.warp_grid(xy_cords, H_invert)

        # change back to ij
        warp_cords = torch.cat((xy_warp_cords[:, :, :, 1].unsqueeze(3), xy_warp_cords[:, :, :, 0].unsqueeze(3)), dim=-1)

        # calc S
        '''
        S[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by 
        the homography is at a distance from (h', w') less than 8 and 0 otherwise
        '''
        cords = cords.view((1, 1, 1, Hc, Wc, 2)).type(torch.float)
        warp_cords = warp_cords.view((B, Hc, Wc, 1, 1, 2))
        distance_map = torch.norm(cords - warp_cords, dim=-1)
        S = distance_map <= 7.5
        S = S.type(torch.float)

        # descriptors
        desc = DESC.view((B, Hc, Wc, 1, 1, -1))
        desc = F.normalize(desc, dim=-1)
        warp_desc = warp_DESC.view((B, 1, 1, Hc, Wc, -1))
        warp_desc = F.normalize(warp_desc, dim=-1)

        # dot product calc
        ''' 
        dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        descriptor at position (h, w) in the original descriptors map and the
        descriptor at position (h', w') in the warped image
        '''
        dot_product = torch.sum(desc * warp_desc, dim=-1)
        relu = torch.nn.ReLU()
        dot_product = relu(dot_product)

        dot_product = F.normalize(dot_product.view((B, Hc, Wc, Hc * Wc)), dim=3)
        dot_product = dot_product.view((B, Hc, Wc, Hc, Wc))

        dot_product = F.normalize(dot_product.view((B, Hc * Wc, Hc, Wc)), dim=1)
        dot_product = dot_product.view((B, Hc, Wc, Hc, Wc))

        # Compute the loss
        pos_margin = 1
        neg_margin = 0.2
        lambda_d = 250
        positive_dist = torch.max(torch.zeros_like(dot_product), pos_margin - dot_product)
        negative_dist = torch.max(torch.zeros_like(dot_product), dot_product - neg_margin)
        loss = lambda_d * S * positive_dist + (1 - S) * negative_dist

        # adjust valid_mask
        block_size = 8
        valid_mask = F.unfold(v_mask, block_size, stride=block_size)
        valid_mask = valid_mask.view(B, block_size ** 2, Hc, Wc)
        valid_mask = valid_mask.permute(0, 2, 3, 1)
        valid_mask = torch.prod(valid_mask, dim=3)
        valid_mask = valid_mask.view((B, 1, 1, Hc, Wc))

        normalization = torch.sum(valid_mask, dim=(1, 2, 3, 4)) * Hc * Wc
        loss = torch.sum(loss * valid_mask, dim=(1, 2, 3, 4)) / normalization
        loss = torch.sum(loss)

        return loss
