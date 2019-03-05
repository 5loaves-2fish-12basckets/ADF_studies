# -*- coding: utf-8 -*-
"""Function for data preparation 

This module includes functions used for data preparation. A typical usage would
be to call make_dataset() and make_dataloader() in trainer.py in when training. 

Example:
    dataset = make_dataset(self.config.data_dir_root, self.args)
    dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)

Todo:
    * enlarge list of datasets to include cifar10, lsun, etc

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import numpy as np
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

#main function to be called

class DataHandler(object):
    def __init__(self, config, args):
        self.datatype = config.datatype
        self.data_dir_root = config.data_dir_root
        self.batch_size = args.batch_size

        self.train_loader = self.build_dataloader(True)
        self.test_loader = self.build_dataloader(False)
        self.shapes = self.get_shapes()

    def build_dataloader(self, train):
        if self.datatype='mnist':
            dataset = self.MNIST(train)
        return data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True) 

    #list of datasets to use
    def MNIST(self, train):
        trans = transforms.Compose([
            # transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

        dataset = datasets.MNIST(
            data_dir_root, train=train, download=True, transform=trans
        )
        return dataset

    def get_shapes(self):
        sample, sample_target = next(iter(self.train_loader))
        data_shape, label_shape = sample[0].shape, sample_target[0].shape

        return (data_shape, label_shape)
