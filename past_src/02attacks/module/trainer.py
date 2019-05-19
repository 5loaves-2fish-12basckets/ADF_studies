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

from module.model import VGG 
from module.datafunc import make_dataloaders

import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import glob

from PIL import Image
from shutil import copyfile

from matplotlib import pyplot as plt

class config():
    def __init__(self):
        self.data_dir_root = '/home/jimmy/datastore'
        self.img_size = 64
        self.batch_size = 128
        self.lr = 0.001
        self.betas = (0.5, 0.999)

class Trainer():
    def __init__(self):
        self.config = config()

        self.model = VGG()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.lr, betas = self.config.betas)

        dataloaders = make_dataloaders(self.config.data_dir_root, 
                                        self.config.img_size,
                                        self.config.batch_size)

        self.trainloader = dataloaders[0]
        self.testloader = dataloaders[1]
        self.fontloader = dataloaders[2]

    def train(self):
        for __ in range(1):
            pbar = tqdm(self.trainloader)
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(targets.view_as(pred)).sum().item()*100//len(pred)

                message = 'loss:%.4f, acc:%d%%'%(loss.item(), acc)
                pbar.set_description(message)

    def run_test(self, dataloader):
        pbar = tqdm(dataloader)
        correct = 0
        total = 0
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            output = self.model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            correct+= pred.eq(targets.view_as(pred)).sum().item()
            total += len(pred)

        acc = correct*100//total
        return acc

    def test(self):
        acc = self.run_test(self.testloader)        
        print('test on mnist test accuracy: %d%%'%acc)

        acc = self.run_test(self.fontloader)        
        print('test on font digit select accuracy: %d%%'%acc)

    @staticmethod
    def fgsm(batchdata, eps, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_data = batchdata + eps*sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data
    
    def run_attack(self, dataloader):
        epsilons = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        accuracies = []
        adv_samples = []
        for eps in epsilons:
            adv_sample = []
            correct = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)    
                inputs.requires_grad = True  
                output = self.model(inputs)
                init_pred = output.argmax(dim=1, keepdim=True)
                if init_pred.item() != targets.item():
                    continue
                
                loss = self.criterion(output, targets)
                self.model.zero_grad()
                loss.backward()

                data_grad = inputs.grad.data
                perturbed_data = self.fgsm(inputs, eps, data_grad)
                pert_output = self.model(perturbed_data)
                final_pred = pert_output.argmax(dim=1, keepdim=True)
                
                if final_pred.item() == targets.item():
                    correct+=1
                    if (eps == 0) and (len(adv_sample) < 5):
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_sample.append( (init_pred.item(), final_pred.item(), adv_ex) )
                else:
                    if len(adv_sample) < 5:
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_sample.append( (init_pred.item(), final_pred.item(), adv_ex) )

            adv_samples.append(adv_sample)
            acc = correct/float(len(dataloader))
            accuracies.append(acc)
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}%".format(eps, correct, len(dataloader), acc))

        return epsilons, accuracies, adv_samples

    @staticmethod
    def plot_eps_acc(epsilons, accuracies, name):
        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.savefig(name)

    @staticmethod
    def save_sample(epsilons, examples, name):
        cnt = 0
        plt.figure(figsize=(8,10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons),len(examples[0]),cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig,adv,ex = examples[i][j]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.savefig(name)

    def attack(self):
        __, testloader, fontloader = make_dataloaders(
            self.config.data_dir_root, self.config.img_size, 1)

        print()
        print('fgsm on mnist test set')
        eps, acc, samples = self.run_attack(testloader)
        self.plot_eps_acc(eps, acc, 'output/fgsm_mnist_acc.png')
        self.save_sample(eps, samples, 'output/fgsm_mnist_samples.png')

        print()
        print('fgsm on font digit select')
        eps, acc, samples = self.run_attack(fontloader)
        self.plot_eps_acc(eps, acc, 'output/fgsm_font_acc.png')
        self.save_sample(eps, samples, 'output/fgsm_font_samples.png')


