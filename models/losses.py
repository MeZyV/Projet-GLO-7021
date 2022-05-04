import torch
import torch.nn as nn
import kornia as K
import torch.nn.functional as F
import numpy as np


class DectectorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true_labels: torch.Tensor, pred: torch.Tensor, v_mask: torch.Tensor = None, device: str = 'cpu'):
        # TODO: Manage device (cause we are creating tensors inside the forward)
        """
        Forward pass compute detector loss

        :param true_labels: Ground truth interest points pytorch tensor shaped N x 1 x H x W.
        :param pred: Dense detector decoder output pytorch tensor shaped N x 65 x Hc x Wc.
        :param v_mask: valid_mask size of true_labels
        :param device: where to load the tensors used in the loss
        :return: Detector loss.
        """
        n, _, h, w = true_labels.size()
        _, c, h_c, w_c = pred.size()
        block_size = 8

        true_labels = true_labels.type(torch.float32)
        # True labels to dense, serves for the fully-convolutional cross-entropy loss
        convolution_labels = F.pixel_unshuffle(true_labels, block_size)
        # Channels = Classes
        convolution_labels = convolution_labels.permute(0, 2, 3, 1)
        # Add dustbin channel to labels (factor 2 to save the labels with the future noise)
        convolution_labels = torch.cat([2 * convolution_labels.to(device), torch.ones((n, h_c, w_c, 1)).to(device)], dim=3)
        # If two ground truth corner positions land in the same bin
        # then we randomly select one ground truth corner location
        # TODO: Way too random
        noise = (torch.rand(convolution_labels.size()) * 0.1).to(device)
        # Get labels
        labels = torch.argmax(convolution_labels + noise, dim=3)

        # Define valid mask
        if v_mask is not None:
            valid_mask = v_mask.type(torch.float32)
            # Adjust valid_mask
            valid_mask = F.pixel_unshuffle(true_labels, block_size)
            valid_mask = valid_mask.permute(0, 2, 3, 1)
            valid_mask = torch.prod(valid_mask, dim=3)
            labels[valid_mask == 0] = 65

        # Get loss
        loss = nn.CrossEntropyLoss(ignore_index=65)
        output = loss(pred, labels)
        return output


class DescriptorLoss(nn.Module):
    def __init__(self, lambda_d=250, pos_margin=1, neg_margin=0.2):
        super().__init__()
        self.lambda_d = lambda_d
        self.pos_margin, self.neg_margin = pos_margin, neg_margin

    def forward(self, desc_, wrap_desc_, H, device: str = 'cpu'):
        """
        Forward pass compute descriptor loss

        :param desc_: Dense descriptor decoder output pytorch tensor shaped N x D x Hc x Wc. (base image)
        :param wrap_desc_: Dense descriptor decoder output pytorch tensor shaped N x D x Hc x Wc. (wrapped image)
        :param H: Homography pytorch tensor shaped N x 3 x 3. (used to wrap the image)
        :param device: where to load the tensors used in the loss
        :return: Descriptor loss.
        """
        # Get the C dimension
        # H = H.unsqueeze(1)

        block_size = 8
        b, d, h_c, w_c = desc_.size()
        desc = desc_.permute(0, 2, 3, 1)
        wrap_desc = wrap_desc_.permute(0, 2, 3, 1)

        # create grid of center of HcxWc region
        coords = torch.stack(torch.meshgrid(torch.arange(0, h_c), torch.arange(0, w_c)), dim=-1).type(
            torch.int32)
        coords = coords.unsqueeze(0)
        coords = coords * block_size + block_size // 2

        # change from ij (yx) to xy coords to warp grid
        xy_coords = torch.cat((coords[:, :, :, 1].unsqueeze(3), coords[:, :, :, 0].unsqueeze(3)), dim=-1)

        # Wrap xy coords

        # List of points homogene
        xy_coords = xy_coords.view(h_c * w_c, 2)
        xy_coords = torch.cat((xy_coords, torch.ones(h_c * w_c, 1)), dim=-1)

        # Apply homography
        # TODO: do it but in torch
        xy_np = xy_coords.permute(1, 0).cpu().numpy()
        h_np = H.cpu().numpy()
        xy_coords_wrap = torch.tensor(np.dot(h_np, xy_np))
        xy_coords_wrap = xy_coords_wrap[:, :2, :] / xy_coords_wrap[:, -1, :].unsqueeze(1)
        xy_coords_wrap = xy_coords_wrap.permute(0, 2, 1).reshape(b, h_c, w_c, 2)

        # change back to ij
        warp_coords = torch.cat((xy_coords_wrap[:, :, :, 1].unsqueeze(3), xy_coords_wrap[:, :, :, 0].unsqueeze(3)),
                                dim=-1)

        # calc S
        # S[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by
        # the homography is at a distance from (h', w') less than 8 and 0 otherwise

        coords = coords.view((1, 1, 1, h_c, w_c, 2)).type(torch.float)
        warp_coords = warp_coords.view((b, h_c, w_c, 1, 1, 2))
        distance_map = torch.norm(coords - warp_coords, dim=-1)
        S = distance_map <= 7.5
        S = S.type(torch.float).to(device)

        # descriptors
        desc = desc.unsqueeze(3).unsqueeze(3)  # (b, h_c, w_c, 1, 1, d)
        desc = F.normalize(desc, dim=-1)
        warp_desc = wrap_desc.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, h_c, w_c, d)
        warp_desc = F.normalize(warp_desc, dim=-1)

        # dot product calc
        ''' 
        dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        descriptor at position (h, w) in the original descriptors map and the
        descriptor at position (h', w') in the warped image
        '''
        dot_product = torch.sum(desc * warp_desc, dim=-1)
        dot_product = F.relu(dot_product)

        dot_product = F.normalize(dot_product.view((b, h_c, w_c, h_c * w_c)), dim=3)
        dot_product = dot_product.view((b, h_c, w_c, h_c, w_c))

        dot_product = F.normalize(dot_product.view((b, h_c * w_c, h_c, w_c)), dim=1)
        dot_product = dot_product.view((b, h_c, w_c, h_c, w_c))

        # Compute the Hinge loss
        pos_margin = self.pos_margin
        neg_margin = self.neg_margin
        lambda_d = self.lambda_d
        positive_dist = torch.max(torch.zeros_like(dot_product), pos_margin - dot_product)
        negative_dist = torch.max(torch.zeros_like(dot_product), dot_product - neg_margin)

        loss = lambda_d * S * positive_dist + (1 - S) * negative_dist

        # Mean loss over h_c w_c h'_c w'_c
        normalization = (h_c * w_c) ** 2
        loss = torch.sum(loss, dim=(1, 2, 3, 4)) / normalization
        loss = torch.sum(loss)

        return loss
