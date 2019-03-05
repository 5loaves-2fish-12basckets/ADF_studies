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

from module.model import MWDNet
from module.datafunc import DataHandler

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
        self.model = MWDNet(args)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD()
        self.config = config
        self.args = args
        self.opt = opt

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.records = []

    def train(self):
        print('training for task:', self.config.taskname)

        dh = DataHandler(self.config, self.args)
        if self.device =='cuda':
            self.model.cuda()

        for i in range(self.opt.epochs):
            epoch_records = self.train_one_epoch(dataloader)
            self.records.append(epoch_records)
            # self.save_img_sample(str(i))
            self.save_fixed_grid_sample(fixed_z,'Epoch_'+str(i))
            self.save_loss_plot('Epoch_'+str(i))
        if self.opt.save_model:
            self.model.save(self.opt.model_filepath)

    def train_one_epoch(self, dataloader):
        real_label = 1
        fake_label = 0

        # pbar = tqdm(enumerate(dataloader))
        pbar = tqdm(dataloader)
        # for index, (inputs, __) in pbar:
        epoch_records = []
        for inputs, __ in pbar:
            batch_size = inputs.shape[0]
            label_real = Variable(torch.ones(batch_size).cuda())
            label_fake = Variable(torch.zeros(batch_size).cuda())

            # Discriminator 
            for __ in range(self.opt.k):
                inputs = inputs.to(self.device)
                inputs = Variable(inputs)
                out = self.model.D(inputs)
                out = out.view(-1)
                err_Dis_real = self.criterion(out, label_real)
                Dis_out = out.mean().item()

                z = torch.randn(batch_size, 100, 1, 1, device=self.device)
                z = Variable(z)

                fake_inputs = self.model.G(z)
                out = self.model.D(fake_inputs)
                out = out.view(-1)
                err_Dis_fake = self.criterion(out, label_fake)

                err_Dis = err_Dis_fake + err_Dis_real
                self.model.D.zero_grad()
                err_Dis.backward()
                self.D_optimizer.step()

            # Generator  maximize log(D(G(z)))
            for __ in range(self.opt.g):
                
                z = torch.randn(batch_size, 100, 1, 1, device=self.device)
                z = Variable(z)
                fake_inputs = self.model.G(z)

                out = self.model.D(fake_inputs)
                out = out.view(-1)
                Dis_gen_out = out.mean().item()

                err_Gen = self.criterion(out, label_real)
                
                self.model.D.zero_grad()
                self.model.G.zero_grad()
                err_Gen.backward()
                self.G_optimizer.step()


            batch_record = (err_Dis.item(), err_Gen.item(), Dis_out, Dis_gen_out)
            epoch_records.append(list(batch_record))
            message = 'errD:%.4f,errG:%.4f,D(x):%.4f,D(G(z)):%.4f'%batch_record
            pbar.set_description(message)

        epoch_records = np.mean(np.array(epoch_records),0)
        return epoch_records.tolist()

    def save_img_sample(self, img_name='generated'):
        with torch.no_grad():
            z = torch.randn(1, 100, 1, 1, device=self.device)
            generated_img = self.model.G(z)
            save_image(generated_img.cpu(), self.opt.img_dir+'/'+img_name+'.png')


    def save_fixed_grid_sample(self, fixed_z, img_name='generated'):
        
        with torch.no_grad():
            fixed_z = Variable(fixed_z)
            generated_ = self.model.G(fixed_z)
            def denorm(x):
                # is this needed???
                out = (x + 1) / 2
                return out.clamp(0, 1)
            genrated_ = denorm(generated_)

            n_rows = np.sqrt(fixed_z.size()[0]).astype(np.int32)
            n_cols = np.sqrt(fixed_z.size()[0]).astype(np.int32)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5,5))
            for ax, img in zip(axes.flatten(), generated_):
                ax.axis('off')
                ax.imshow(img.cpu().data.view(self.args.img_size, self.args.img_size).numpy(), cmap='gray', aspect='equal')
            plt.subplots_adjust(wspace=0, hspace=0)
            # title = 'Epoch {0}'.format(num_epoch+1)
            fig.text(0.5, 0.04, img_name, ha='center')

            filepath = self.opt.task_dir+'/generated_imgs'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            plt.savefig(filepath+'/'+img_name+'.png')
            plt.close()

    def save_loss_plot(self, img_name='loss'):
        four_records = np.array(self.records).T
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.opt.epochs)
        ax.set_xlabel(img_name)

        ax.set_ylim(0, np.max(four_records[:2])*1.1)
        ax.set_ylabel('Loss values')
        ax.plot(four_records[0], label='D loss', color='#CC0000')
        ax.plot(four_records[1], label='G loss', color='#FF8000')
        ax.tick_params('y', colors = '#CC0000')
        ax.legend(loc='upper left')

        ax2 = ax.twinx()
        ax2.set_ylim(0, np.max(four_records[2:])*1.1)
        ax2.set_ylabel('D() values')
        ax2.plot(four_records[2], label='D(x)', color='#0080FF', linestyle=':')
        ax2.plot(four_records[3], label='D(G(z))', color='#808080', linestyle=':')
        ax2.tick_params('y', colors='k')
        ax2.legend(loc='upper right')

        
        filepath = self.opt.task_dir+'/loss_plots'
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        plt.savefig(filepath+'/'+img_name+'.png')
        plt.close()




