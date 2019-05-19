# -*- coding: utf-8 -*-
"""Function for basic model classes
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch
import torch.nn as nn

def Linear_model():
    model = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28*28, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
        # torch.nn.Softmax()
        )
    return model

class Flatten(torch.nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

## 421 half size 311 same size

def Wide_model(k=5):
    model = nn.Sequential(
        nn.Conv2d(1, 4*k, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*k*7*7,k*128),
        nn.ReLU(),
        nn.Linear(k*128, 10)
    )
    return model


def Deep_model(k=2):
    def conv_group(inf, outf, N): 
        model_list = []
        for i in range(N):
            if i == 0:
                c_in = inf
                c_out = outf
            else:
                c_in = c_out = outf

            if i == N-1:
                model_list.append(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1))
            else:
                model_list.append(nn.Conv2d(c_in, c_out, 3, stride=1, padding=1))
            model_list.append(nn.ReLU())

        return model_list

    conv_group1 = conv_group(1, 8, k)
    conv_group2 = conv_group(8, 16, k)

    model = nn.Sequential(
        *conv_group1, 
        *conv_group2,
        Flatten(),
        nn.Linear(16*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# def VGG():


# def ResNet():
#     pass










