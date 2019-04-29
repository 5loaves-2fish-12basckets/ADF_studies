# -*- coding: utf-8 -*-
"""Function for basic model classes
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import torch

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
# def Wide_model():
#     pass

# def Deep_model():
#     pass

# def VGG():
#     pass

# def ResNet():
#     pass














# # basic models

# def model_wide(in_ch, out_width, k): 
#     model = nn.Sequential(
#         nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
#         nn.ReLU(),
#         Flatten(),
#         nn.Linear(8*k*out_width*out_width,k*128),
#         nn.ReLU(),
#         nn.Linear(k*128, 10)
#     )
#     return model

# def model_deep(in_ch, out_width, k, n1=8, n2=16, linear_size=100): 
#     def group(inf, outf, N): 
#         if N == 1: 
#             conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), 
#                          nn.ReLU()]
#         else: 
#             conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), 
#                          nn.ReLU()]
#             for _ in range(1,N-1):
#                 conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
#                 conv.append(nn.ReLU())
#             conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
#             conv.append(nn.ReLU())
#         return conv

#     conv1 = group(in_ch, n1, k)
#     conv2 = group(n1, n2, k)


#     model = nn.Sequential(
#         *conv1, 
#         *conv2,
#         Flatten(),
#         nn.Linear(n2*out_width*out_width,linear_size),
#         nn.ReLU(),
#         nn.Linear(100, 10)
#     )
#     return model

# # typical models

# # Linear
# # Conv
# # VGG
# # ResNet
