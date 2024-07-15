'''
This file is copied from
https://github.com/yulunzhang/RCAN
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a default 2D convolution layer with specified input/output channels, kernel size, and bias
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# Mean shift layer to normalize images based on given mean and standard deviation values
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)  # Convert standard deviations to tensor
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)  # Initialize weight as an identity matrix
        self.weight.data.div_(std.view(3, 1, 1, 1)) # Divide weights by standard deviations
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) # Set bias based on mean values and sign
        self.bias.data.div_(std) # Divide bias by standard deviations
        self.requires_grad = False # Do not update weights during training

# Basic block consisting of convolution, batch normalization, and activation layers
class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))  # Add batch normalization if specified
        if act is not None:
            m.append(act)  # Add activation layer if specified
        # Initialize using the parent class with the list of layers
        super(BasicBlock, self).__init__(*m)

# Residual block with two convolution layers and an optional scaling factor
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            # Append convolution layers
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                # Optionally append batch normalization layer
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                # Add activation only after the first convolution
                m.append(act)
        # Combine all layers into a sequential module
        self.body = nn.Sequential(*m)
        # Set residual scaling factor
        self.res_scale = res_scale

    def forward(self, x):
        # Forward pass through the residual block, applying scaling
        res = self.body(x).mul(self.res_scale)
        # Residual connection adds input to the output
        res += x

        return res

# Upsampling block to increase the image resolution by the given scale
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        # If the scale is a power of 2, repeatedly apply pixel shuffle
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        # Handle a scaling factor of 3
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        # Raise an error if the scale factor is not supported
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
