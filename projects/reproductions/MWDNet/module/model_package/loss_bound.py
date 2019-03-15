#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# LOSS = {
#     'nll': F.nll_loss,
#     'sdl': square_distance_loss,
#     'sdl_soft': square_distance_loss_soft,
#     'sdl_distr': square_distance_loss_distr,
# }

def square_distance_loss(output, target):
    """Returns the square distacne loss between output and target, except that
    the real reference output is a vector with all 0, with a 1 in the position
    specified by target."""
    s = list(output.shape)
    n_classes = s[-1]
    out = output.view(-1, n_classes)
    ss = out.shape
    n_els = ss[0]
    idxs = target.view(-1)
    t = output.new(n_els, n_classes)
    t.requires_grad = False
    t.fill_(0.)
    t[range(n_els), idxs] = 1.
    d = out - t
    dd = d * d
    return torch.sum(dd) / n_els


def square_distance_loss_soft(output, target, non_zero=0.1):
    """Like square distance loss between output and a vector-valued
    one-hot target, except that one-hot vector is encoded as non_zero's and
    1 minus non_zero.  Since the vector outputs are normalized, this discourages
    the appearance of progress by making the "correct" output overly large."""
    s = list(output.shape)
    n_classes = s[-1]
    out = output.view(-1, n_classes)
    ss = out.shape
    n_els = ss[0]
    idxs = target.view(-1)
    t = output.new(n_els, n_classes)
    t.requires_grad = False
    t.fill_(0.)
    t[range(n_els), idxs] = 1.
    t = t * (1.0 - non_zero) + non_zero / float(n_classes)
    d = out - t
    dd = d * d
    return torch.sum(dd) / n_els


def square_distance_loss_distr(output, target, non_zero=0.1):
    """This version is similar to square_disance_loss_soft, except that
    it first computes a distribution from the output values, that have to
    be non-negative."""
    distr = output / (torch.sum(output, -1, keepdim=True) + 0.001)
    return square_distance_loss_soft(distr, target, non_zero=non_zero)


class BoundedParameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.
    This constructor returns a Parameter, but adds definable upper and lower bounds.
    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
        lower_bound (float, optional): lower bound for the values.
            This is observed only by bounds-aware optimizers.
        upper_bound (float, optional): upper bound for the values.
            This is observed only by bounds-aware optimizers.
    """
    def __new__(cls, data=None, requires_grad=True, lower_bound=None, upper_bound=None):
        if data is None:
            data = torch.Tensor()
        p = torch.Tensor._make_subclass(Parameter, data, requires_grad)
        p.lower_bound = lower_bound
        p.upper_bound = upper_bound
        return p


class ParamBoundEnforcer(object):
    """Wrapper for any optimizer that enforces parameter bounds."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def enforce(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if hasattr(p, 'lower_bound'):
                    lower = getattr(p, 'lower_bound')
                    upper = getattr(p, 'upper_bound')
                    if lower is not None or upper is not None:
                        p.data.clamp_(min=lower, max=upper)

