import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict


class SuperPointNet(torch.nn.Module):
    def __init__(self, superpoint_bool):
        super(SuperPointNet, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1a', nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv1b', nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2a', nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv2b', nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3a', nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv3b', nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4a', nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv4b', nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)),
            ('relu8', nn.ReLU(inplace=True))
        ]))
        # Detector Head.
        self.detector = nn.Sequential(OrderedDict([
            ('convPa', nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)),
            ('relu', nn.ReLU(inplace=True)),
            ('convPb', nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)),
        ]))
        # Descriptor Head.
        self.descriptor = nn.Sequential(OrderedDict([
            ('convDa', nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)),
            ('relu', nn.ReLU(inplace=True)),
            ('convDb', nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)),
        ]))
        self.superpoint_bool = superpoint_bool


    def forward(self, x, dense=True):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
            x: Image pytorch tensor shaped N x 1 x H x W.
        Output
            semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
            desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8(super) 
            or None(magicpoint).
        """
        # Shared Encoder.
        x = self.encoder(x)

        # Detector Head.
        semi = self.detector(x)

        # if we want a heatmap the size of the input image.
        if not dense:
            # Softmax.
            # "channel-wise Softmax" non-learned transformation
            # Not used to compute loss
            dense = F.softmax(semi, 1)
            # Remove dustbin.
            nodust = dense[:, :-1, :, :]
            # Upsampling
            semi = F.pixel_shuffle(nodust, 8)

        # Descriptor Head.
        # if we want superpoint model:
        if self.superpoint_bool:
            desc = self.descriptor(x)
            # if we want a descriptor the size of the input image.
            if not dense:
                dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
                desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # if we want magicpoint model:
        else:
            desc = None

        return semi, desc