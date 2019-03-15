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

import sys
sys.path.append('module')
from datafunc import DataHandler
from model_package.mwdnet import MWDNet
from model_package.loss_bound import square_distance_loss, ParamBoundEnforcer


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
        self.datahandler = DataHandler(config, args)
        datashape, __ = self.datahandler.get_shapes()
        init_channel = datashape[0]*datashape[-1]*datashape[-2]
        # prepare model 

        self.config = config
        self.args = args
        self.opt = opt


        self.model = MWDNet(args, init_channel)
        self.criterion = square_distance_loss
        self.meta_optimizer = self.create_optimizer()

        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        print('training for task:', self.config.taskname)

        # prepare dataset dataloader is dh.x_loader x=train/test

        if self.device =='cuda':
            self.model.cuda()

        for i in range(self.opt.epochs):
            self.train_one_epoch(self.datahandler.train_loader)

        if self.opt.save_model:
            self.model.save(self.opt.model_filepath)

    def train_one_epoch(self, dataloader):

        pbar = tqdm(dataloader)

        for inputs, targets in pbar:
            correct = 0
            batch_size = inputs.shape[0]

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.meta_optimizer.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.meta_optimizer.optimizer.step()
            self.meta_optimizer.enforce() 


            pred = outputs.max(1)[1]
            correct= pred.eq(targets).sum().item()
            # correct+= pred.eq(targets.vew_as(pred)).sum().item()

            epoch_record = (loss.item(), 100.*correct/len(targets), self.model.sensitivity().item())
            message = 'Loss:%.4f,Acc:%.d,Sensitivity:%.4f'%epoch_record
            pbar.set_description(message)

    def create_optimizer(self):
    # Creates an optimizer.
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        # if args.opt == 'mom':
        #     optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
        # elif args.opt == 'adam':
        #     optimizer = optim.Adam(params, lr=args.lr)
        # else:
        optimizer = optim.Adadelta(params, lr=self.args.lr)
        # For MWD nets, we wrap it into an enforcer.
        meta_optimizer = ParamBoundEnforcer(optimizer) # if model.bounded_parameters else optimizer
        return meta_optimizer

