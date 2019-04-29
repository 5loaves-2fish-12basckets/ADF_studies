# -*- coding: utf-8 -*-
"""Function for dataset and dataloader preparation
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""
import torch
import torchvision

def make_dataloaders(data_dir_root='/home/jimmy/datastore', img_size=28, batch_size=128):

    mnist_trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ])
    trainset = torchvision.datasets.MNIST(data_dir_root, train=True, download=False, transform=mnist_trans)
    testset = torchvision.datasets.MNIST(data_dir_root, train=False, download=False, transform=mnist_trans)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader
