# -*- coding: utf-8 -*-
"""Main training module for pytorch training on DCGAN model
This module puts together data and model, and perform various DCGAN 
trainings, including train, 
    
Example:
    trainer = Trainer(config, args, opt)
    trainer.train()

Todo:
    arithmetic operations, other functions needed for experiment
    show training history
    gif result
    grid result
    evaluation???
    

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.model import ResNet as model
# from module.model import VGG as model
from module.datafunc import make_dataloader

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable


from tqdm import tqdm
from random import randint
from torchvision.utils import save_image
from matplotlib import pyplot as plt 
plt.switch_backend('agg')

import numpy as np

import os

class Trainer():
    def __init__(self, config, args, opt):
        self.model = model()
        self.optimizer =  optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.args = args
        self.opt = opt

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.records = []

        self.trainloader, self.testloader = make_dataloader(self.config.data_dir_root, 
                                        self.args.img_size,
                                        self.args.batch_size)
        if self.device =='cuda':
            self.model.cuda()

    def train(self):
        print('training for task:', self.config.taskname)

        print('train %d epochs'%self.opt.epochs)
        # for i in range():
        for i in range(self.opt.epochs):
            self.train_one_epoch()
        return self.test()

    def train_one_epoch(self):
        pbar = tqdm(self.trainloader)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            inputs = Variable(inputs, requires_grad = True)
            targets = targets.to(self.device)
            targets = Variable(targets)
            output = self.model(inputs)
            loss = self.criterion(output, targets)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(targets.view_as(pred)).sum().item()*100//len(pred)
            
            message = 'loss:%.4f, accuracy: %d%%'%(loss.item(), accuracy)            
            pbar.set_description(message)

    def test(self):
        pbar = tqdm(self.testloader)
        accuracy = 0
        total = 0
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            output = self.model(inputs)
            pred =  output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(targets.view_as(pred)).sum().item()
            total += len(pred)

        accuracy = accuracy/total * 100

        message = 'test accuracy: %d'%accuracy
        print(message)
        return accuracy





