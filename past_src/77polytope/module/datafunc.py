# -*- coding: utf-8 -*-
"""Function for dataset and dataloader preparation
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""
import torch
import torchvision

def make_dataloaders(data_dir_root='/home/jimmy/datastore', img_size=28, batch_size=128, mode=None):

    mnist_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])

    affine_trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine((0.1,0.2), translate=(0.1,0.1), scale=(1.1, 1.2)),
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])
    
    color_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8,1.2), saturation=(0.8, 1.2), hue=0.1),
            torchvision.transforms.ToTensor(),
        ])
    if mode is None:
        ttrans = mnist_trans
    elif mode == 'aff':
        ttrans = affine_trans
    elif mode == 'col':
        ttrans = color_trans

    trainset = torchvision.datasets.MNIST(data_dir_root, train=True, download=False, transform=mnist_trans)
    testset = torchvision.datasets.MNIST(data_dir_root, train=False, download=False, transform=ttrans)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
