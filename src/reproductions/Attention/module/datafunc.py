# -*- coding: utf-8 -*-
"""Function for data preparation 

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from torch.utils import data
from torchvision import datasets
from torchvision import transforms

#main function to be called
def make_dataloader(data_dir_root, datatype, img_size, batch_size):
    if datatype=='mnist':
        dataset = MNIST(data_dir_root, img_size, True)
        train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = MNIST(data_dir_root, img_size, False)
        test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif datatype=='cifar10':
        dataset = CIFAR10(data_dir_root, img_size, True)
        train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = CIFAR10(data_dir_root, img_size, False)
        test_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

#list of datasets to use
def MNIST(data_dir_root, img_size, train):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset = datasets.MNIST(
        data_dir_root, train=train, download=True, transform=trans
    )
    return dataset

def CIFAR10(data_dir_root, img_size, train):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset = datasets.CIFAR10(
        data_dir_root, train=train, download=True, transform=trans
    )
    return dataset
