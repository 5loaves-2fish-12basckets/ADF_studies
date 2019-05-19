# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from dann_module.dsnmodel import DSN, MSE, scale_inv_MSE, DifferenceLoss
from dann_module.datafunc import make_dataloaders, mnist_dataloaders
import torch
from tqdm import tqdm
import numpy as np
import os

torch.manual_seed(7)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True 

class training_options(object):
    def __init__(self):
        self.lr = 0.01
        self.step_decay_weight = 0.95
        self.lr_decay_step = 20000
        # self.activate_domain_loss_step = 1000
        self.activate_domain_loss_step = 10000
        self.activate_domain_epoch = 5
        self.weight_decay = 1e-6
        self.alpha_weight = 0.01
        # self.beta_weight = 0.1
        self.beta_weight = 0.075
        self.gamma_weight = 0.25
        # self.gamma_weight = 1
        self.momentum = 0.9

    def lr_scheduler(self, optimizer, step):
        current_lr = self.lr * (self.step_decay_weight ** (step / self.lr_decay_step))
        if step % self.lr_decay_step == 0:
            print('learning rate is set to %f'%current_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        return optimizer

class Trainer():
    def __init__(self, args):
        args.z_dim = 100
        args.n_classes = 10
        self.model = DSN(args)
        # self.optimizer = torch.optim.Adam(self.model.parameters())

        self.class_loss = torch.nn.CrossEntropyLoss()
        self.rec_loss = MSE()
        self.rec_loss2 = scale_inv_MSE()
        self.diff_loss = DifferenceLoss()
        self.simi_loss = torch.nn.CrossEntropyLoss()

        dataloaders = make_dataloaders(args.source, args.target, args.batch_size)
        # dataloaders = mnist_dataloaders(args.batch_size)
        self.sourceloader, self.targetloader, self.testtargetloader = dataloaders

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        object_list = [self.model, self.class_loss, self.rec_loss, self.rec_loss2, self.diff_loss, self.simi_loss]
        for object_ in object_list:
            object_.to(self.device)

        # for parameter in self.model.parameters():
        #     parameter.requires_grad = True

        self.args = args
        self.modelpath = os.path.join('ckpt', args.taskname, 'model_%s.pth'%args.target[:2])
        self.bestpath = os.path.join('ckpt', args.taskname, 'best_%s.pth'%args.target[:2])
        print(self.modelpath)
        print(self.bestpath)

        self.opt = training_options()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.opt.lr, 
                                        momentum=self.opt.momentum, 
                                        weight_decay=self.opt.weight_decay)



    def train(self):
        print('%s --> %s'%(self.args.source, self.args.target))
        best_acc = 0
        step = 0
        bbar = tqdm(range(self.args.epochs), ncols=100)
        # for epoch in range(self.args.epochs):
        for epoch in bbar:
            self.model.train()
            step = self.train_one_epoch(epoch, step)
            self.model.eval()
            acc = self.test()
            torch.save(self.model.state_dict(), self.modelpath)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), self.bestpath)
            bbar.set_postfix(acc=acc, best_acc=best_acc)

    def train_one_epoch(self, epoch, step):

        num_iteration = min(len(self.sourceloader), len(self.targetloader))
        dann_epoch = int(self.opt.activate_domain_loss_step / num_iteration)
        pbar = tqdm(range(num_iteration), ncols=100, desc=str(epoch))
        
        for i in pbar:

            #train with target data
            self.model.zero_grad()
            loss=0
            timg, __ = next(iter(self.targetloader))
            timg = timg.to(self.device)
            batch_size = len(timg)
            domain_label = torch.ones(batch_size).long().to(self.device)
            dom_acc = 0
            sdom_acc = 0
            alpha=0
            p=0
            # if step > self.opt.activate_domain_loss_step:
            if epoch > self.opt.activate_domain_epoch:
                dann_epoch = self.opt.activate_domain_epoch
                p = float(i + (epoch - dann_epoch) * num_iteration) / (self.args.epochs - dann_epoch) / num_iteration
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                result = self.model(timg, mode='target', alpha=alpha)
                targ_priv_code, targ_share_code, targ_dom_lab, targ_rec = result
                target_dann = self.opt.gamma_weight * self.simi_loss(targ_dom_lab, domain_label)
                dom_acc = targ_dom_lab.argmax(1).eq(domain_label).float().mean().item()
                loss += target_dann

            else:
                target_dann = torch.zeros(1)  # just to fill place
                result = self.model(timg, mode='target')
                targ_priv_code, targ_share_code, __, targ_rec = result

            targ_diff = self.opt.beta_weight * self.diff_loss(targ_priv_code, targ_share_code)
            # targ_mse = self.opt.alpha_weight * self.rec_loss(targ_rec, timg)
            targ_simse = self.opt.alpha_weight * self.rec_loss2(targ_rec, timg)
            loss += (targ_diff + targ_simse)
            # loss += (targ_diff + targ_mse + targ_simse)

            # self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # loss_t = loss.item()
            self.optimizer.step()

            # train with source data
            self.model.zero_grad()
            loss = 0
            simg, slabel = next(iter(self.sourceloader))
            simg, slabel = simg.to(self.device), slabel.to(self.device)
            batch_size = len(simg)
            domain_label = torch.zeros(batch_size).long().to(self.device)

            # if step > self.opt.activate_domain_loss_step:
            if epoch > self.opt.activate_domain_epoch:
                result = self.model(simg, mode='source', alpha=alpha)
                sour_priv_code, sour_share_code, sour_dom_lab, class_label, sour_rec = result
                
                source_dann = self.opt.gamma_weight * self.simi_loss(sour_dom_lab, domain_label)
                sdom_acc = sour_dom_lab.argmax(1).eq(domain_label).float().mean().item()
                loss += source_dann

            else:
                source_dann = torch.zeros(1)
                result = self.model(simg, mode='source')
                sour_priv_code, sour_share_code, __, class_label, sour_rec = result

            class_ = self.class_loss(class_label, slabel)
            sour_diff = self.opt.beta_weight * self.diff_loss(sour_priv_code, sour_share_code)
            # sour_mse = self.opt.alpha_weight * self.rec_loss(sour_rec, simg)
            sour_simse = self.opt.alpha_weight * self.rec_loss2(sour_rec, simg)
            loss += (class_ + sour_diff + sour_simse)
            # loss=class_
            # loss += (class_ + sour_diff + sour_mse + sour_simse)
            # print('code?')
            # print(sour_priv_code[0][:10])
            # print(sour_share_code[0][:10])
# 
            # print(class_label.shape)
            # print(class_label[0])
            # print(class_label.argmax(dim=1).cpu().numpy())
            # print(slabel)
            # input()
            class_acc = class_label.argmax(1).eq(slabel).sum().item()*100//len(slabel)
            # self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            step += 1
            
            pbar.set_postfix(acc=class_acc, sda=sdom_acc, tda=dom_acc, alpha=alpha, p=p)

        return step

    def test(self):
        correct = 0
        length = 0
        # pbar = tqdm(self.testtargetloader, ncols=100, desc=self.args.target)
        for images, labels in self.testtargetloader:
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = len(images)
            result = self.model(images, mode='source')
            pred = result[3].argmax(1)
            # pred = result[3].argmax(1, keepdim=True)
            correct += pred.eq(labels).sum().item()
            length += batch_size
        accuracy = correct *100//length
        return accuracy