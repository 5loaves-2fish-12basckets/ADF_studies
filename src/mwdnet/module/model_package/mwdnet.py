# -*- coding: utf-8 -*-
"""Function for DCGAN model class delaration
This module includes the DCGAN model

Example:
    model = GAN(args)
    # args should contain z_dim, layer_G, layer_D, use_batchnorm, use_relu
    model.G
    model.D

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MWDNet(nn.Module):
    def __init__(self, args)
        super(MWDNet, self).__init__()
        self.layer_sizes ### layers + args.n_classes
        self.layers = nn.ModuleList()
        in_channel = args.input_channes * args.input_y_size * args.input_x_size ##this is datasape
        for i, channel_size in enumerate(self.layer_sizes):
            l = unit(in_channel, channel_size,
                andor=args.andor[i],
                min_slope=args.min_slope,
                max_slope=args.max_slope)
            self.layers.append(l)
            in_channel = channel_size

    def forward(self, x):
        x = x.flatten(1)
        for l in self.layers:
            x = l(x)
        return x

    def interval_forward(self, x_min, x_max):
        x_min, x_max = x_min.flatten(1), x_max.flatten(1)
        for l in self.layers:
            x_min, x_max = l.interval_forward(x_min, x_max)

        return x_min, x_max

    def sensitivity(self):
        s = None
        for l in self.layers:
            s = l.sensitivity(s)

        return torch.max(s)


class unit(nn.Module):
    def __init__(self, in_features, out_features, andor="*")
