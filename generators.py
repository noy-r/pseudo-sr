import torch
import torch.nn as nn

from . import common

import sys
#sys.path.append('/Users/noymachluf/Desktop/pseudo-sr/models/common')
#import common


class TransferNet(nn.Module):
    def __init__(self, n_feat=64, z_feat=8, leaky_neg=0.2, n_resblock=6, bn=True, rgb_range=255, rgb_mean=(0.5, 0.5, 0.5),):
        super(TransferNet, self).__init__()
        # Standard deviation values for each RGB channel
        rgb_std = (1.0, 1.0, 1.0)
        # Subtract mean to normalize inputs
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        # Add mean to denormalize outputs
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        # LeakyReLU negative slope
        leaky_neg = leaky_neg
        filter_size = 5
        # Channels for the latent variable `z`
        z_channel = z_feat
        # Layers for processing the image input
        in_img = [common.default_conv(3, n_feat//2, filter_size)]
        if bn:
            # Add Batch Normalization if specified
            in_img.append(nn.BatchNorm2d(n_feat//2))
        in_img.append(nn.LeakyReLU(leaky_neg))
        in_img.append(common.ResBlock(common.default_conv, n_feat//2, filter_size, bn=bn, act=nn.LeakyReLU(leaky_neg)))
        # Sequential model to process image input
        self.img_head = nn.Sequential(*in_img)
        # Layers to process the latent input `z`
        in_z = [nn.ConvTranspose2d(1, z_channel, 2, 2, 0, 0), # 8 -> 16
                nn.LeakyReLU(leaky_neg),
                nn.ConvTranspose2d(z_channel, 2 * z_channel, 2, 2, 0, 0), # 16 -> 32
                nn.LeakyReLU(leaky_neg)]
        # Sequential model to process latent input
        self.z_head = nn.Sequential(*in_z)
        # Combine image and latent features
        self.merge = nn.Conv2d(n_feat//2 + 2*z_channel, n_feat, 1, 1, 0)
        # Residual blocks for feature refinement
        resblocks = [
            common.ResBlock(common.default_conv, n_feat, filter_size, bn=bn, act=nn.LeakyReLU(leaky_neg)) \
            for _ in range(n_resblock)]
        # Sequential model for residual blocks
        self.res_blocks = nn.Sequential(*resblocks)
        # Fusion layers to refine features before output
        self.fusion = nn.Sequential(
            common.default_conv(n_feat, n_feat//2, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//2, n_feat//4, 1),
            nn.LeakyReLU(leaky_neg),
            common.default_conv(n_feat//4, 3, 1))

    def forward(self, x, z=None):
        # Normalize input image
        out_x = self.sub_mean(x)
        # Pass image features through the `img_head`
        out_x = self.img_head(out_x)
        # Pass latent features through the `z_head`
        out_z = self.z_head(z)
        # Merge image and latent features
        out = self.merge(torch.cat((out_x, out_z), dim=1))
        # Pass merged features through residual blocks
        out = self.res_blocks(out)
        # Refine features using the fusion layer
        out = self.fusion(out)
        # Denormalize output image
        out = self.add_mean(out)
        return out

if __name__ == "__main__":
    rgb_range = 1
    rgb_mean = (0.0, 0.0, 0.0)
    model = TransferNet(n_feat=64, n_resblock=5, bn=False, rgb_range=rgb_range, rgb_mean=rgb_mean)
    print(model)

    X = torch.rand(2, 3, 32, 32, dtype=torch.float32) * rgb_range
    Z = torch.randn(2, 1, 8, 8, dtype=torch.float32)
    Y = model(X, Z).detach()
    print(X.shape, Y.shape)
    print(Y.min(), Y.max())
