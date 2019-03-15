# -*- coding: utf-8 -*-
"""
This module rewrites MWDNet

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import Function

from model_package.loss_bound import BoundedParameter


class MWDNet(nn.Module):
    def __init__(self, args, init_channel):
        super(MWDNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channel = init_channel
        for i, channel_size in enumerate(args.layer_sizes):
            l = unit(in_channel, channel_size,
                andor=args.andor[i])
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

    # can this work?
    def save(self, filepath): 
        state = {
            'net': self.layers.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.layers.load_state_dict(state['net'])


class LargeAttractorExp(Function):
    """Implements e^-x with soft derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(-x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return - grad_output / torch.sqrt(1. + x)


class SharedFeedbackMax(Function):

    @staticmethod
    def forward(ctx, x):
        y, _ = torch.max(x, -1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        y_complete = y.view(list(y.shape) + [1])
        d_complete = grad_output.view(list(grad_output.shape) + [1])
        return d_complete * torch.exp(x - y_complete)


class unit(nn.Module):
    def __init__(self, in_features, out_features, andor="*", 
        min_input=0.0, max_input=0.0, min_slope=0.001, max_slope=10.0):
        super(unit, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.andor = andor
        self.modinf = True #modinf
        self.regular_deriv = False
        self.w = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_input, upper_bound=max_input)
        self.u = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_slope, upper_bound=max_slope)
        if andor == 'v':
            self.andor01 = Parameter(torch.ones((1, out_features)))
        elif andor == '^':
            self.andor01 = Parameter(torch.zeros((1, out_features)))
        else:
            self.andor01 = Parameter(torch.Tensor(1, out_features))
            self.andor01.data.random_(0, 2)
        self.andor01.requires_grad = False
        self.w.data.uniform_(min_input, max_input)
        # Initialization of u.
        self.u.data.uniform_(0.2, 0.7)  # These could be parameters.
        self.u.data.clamp_(min_slope, max_slope)


    def forward(self, x):
        # Let n be the input size, and m the output size.
        # The tensor x is of shape * n. To make room for the output,
        # we view it as of shape * 1 n.
        # Aggregates into a modulus.
        xx = x.unsqueeze(-2)
        xuw = self.u * (xx - self.w)
        xuwsq = xuw * xuw
        if self.modinf:
            # We want to get the largest square, which is the min one as we changed signs.
            if self.regular_deriv:
                z, _ = torch.max(xuwsq, -1)
                y = torch.exp(- z)
            else:
                z = SharedFeedbackMax.apply(xuwsq)
                y = LargeAttractorExp.apply(z)
        else:
            z = torch.sum(xuwsq, -1)
            if self.regular_deriv:
                y = torch.exp(- z)
            else:
                y = LargeAttractorExp.apply(z)
        # Takes into account and-orness.
        if self.andor == '^':
            return y
        elif self.andor == 'v':
            return 1.0 - y
        else:
            return y + self.andor01 * (1.0 - 2.0 * y)


    def interval_forward(self, x_min, x_max):
        xx_min = x_min.unsqueeze(-2)
        xx_max = x_max.unsqueeze(-2)
        xuw1 = self.u * (xx_min - self.w)
        xuwsq1 = xuw1 * xuw1
        xuw2 = self.u * (xx_max - self.w)
        xuwsq2 = xuw2 * xuw2
        sq_max = torch.max(xuwsq1, xuwsq2)
        sq_min = torch.min(xuwsq1, xuwsq2)
        # If w is between x_min and x_max, then sq_min should be 0.
        # So we multiply sq_min by something that is 0 if x_min < w < x_max.
        sq_min = sq_min * ((xx_min > self.w) + (self.w > xx_max)).float()

        y_min = torch.exp(- torch.max(sq_max, -1)[0])
        y_max = torch.exp(- torch.max(sq_min, -1)[0])
        # Takes into account and-orness.
        if self.andor == '^':
            return y_min, y_max
        elif self.andor == 'v':
            return 1.0 - y_max, 1.0 - y_min
        else:
            y1 = y_min + self.andor01 * (1.0 - 2.0 * y_min)
            y2 = y_max + self.andor01 * (1.0 - 2.0 * y_max)
            y_min = torch.min(y1, y2)
            y_max = torch.max(y1, y2)
            return y_min, y_max


    def overall_sensitivity(self):
        """Returns the sensitivity to adversarial examples of the layer."""
        if self.modinf:
            s = torch.max(torch.max(self.u, -1)[0], -1)[0].item()
        else:
            s = torch.max(torch.sqrt(torch.sum(self.u * self.u, -1)))[0].item()
        s *= np.sqrt(2. / np.e)
        return s


    def sensitivity(self, previous_layer):
        """Given the sensitivity of the previous layer (a vector of length equal
        to the number of inputs), it computes the sensitivity to adversarial examples
         of the current layer, as a vector of length equal to the output size of the
         layer.  If the input sensitivity of the previous layer is None, then unit
         sensitivity is assumed."""
        if previous_layer is None:
            previous_layer = self.w.new(1, self.in_features)
            previous_layer.fill_(1.)
        else:
            previous_layer = previous_layer.view(1, self.in_features)
        u_prod = previous_layer * self.u
        if self.modinf:
            # s = torch.max(u_prod, -1)[0]
            s = SharedFeedbackMax.apply(u_prod)
        else:
            s = torch.sqrt(torch.sum(u_prod * u_prod, -1))
        s = s * np.sqrt(2. / np.e)
        return s


    def dumps(self):
        """Writes itself to a string."""
        # Creates a dictionary
        d = dict(
            in_features=self.in_features,
            out_features=self.out_features,
            min_input=self.w.lower_bound,
            max_input=self.w.upper_bound,
            min_slope=self.u.lower_bound,
            max_slope=self.u.upper_bound,
            modinf=self.modinf,
            regular_deriv=self.regular_deriv,
            andor=self.andor,
            andor01=self.andor01.cpu().numpy(),
            u=self.u.data.cpu().numpy(),
            w=self.w.data.cpu().numpy(),
        )
        return Serializable.dumps(d)


