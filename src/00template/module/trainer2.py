# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.model import DaNN
from module.datafunc import make_dataloaders
import torch
from tqdm import tqdm
import numpy as np
import os

from convex_adversarial import robust_loss, robust_loss_parallel


torch.manual_seed(7)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True 

class Trainer():
    def __init__(self, args):
        self.model = DaNN()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_domain = torch.nn.CrossEntropyLoss()
        dataloaders = make_dataloaders(args.source, args.target, args.batch_size)
        self.sourceloader = dataloaders[0]
        self.sourcetestloader = dataloaders[1]
        self.targetloader = dataloaders[2]
        self.targettestloader = dataloaders[3]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for to_cuda_obj in [self.model, self.criterion, self.criterion_domain]:
            to_cuda_obj.to(self.device)
        
        self.cert = args.cert
        self.args = args

        stot = (self.args.source[:1], self.args.target[:1])
        self.modelpath = os.path.join('ckpt', self.args.taskname, 'model_%s_%s.pth'%stot)
        self.certpath = os.path.join('ckpt', self.args.taskname, 'cert_%s_%s.pth'%stot)


    def train(self):
        print('%s --> %s'%(self.args.source, self.args.target))
        print('half')
        best_acc = 0
        # bbar = range(self.args.epochs)
        if self.args.resume is None:
            bbar = tqdm(range(self.args.epochs), ncols=100)
            for epoch in bbar:
                self.model.train()
                self.train_one_epoch(epoch)
                self.model.eval()
                sacc, acc = self.test()
                if acc > best_acc:
                    best_acc = acc
                    if self.cert:
                        torch.save(self.model.state_dict(), self.certpath)
                    else:
                        torch.save(self.model.state_dict(), self.modelpath)
                # print(sacc, acc, best_acc)
                bbar.set_postfix(acc=acc, sacc=sacc, best_acc=best_acc)
        modelpath = self.certpath if self.cert else self.modelpath
        self.model.load_state_dict(torch.load(modelpath))
        result = self.attack()
        print('source fgsm pgd')
        print(result[0][0])
        print(result[0][1])
        print('target fgsm pgd')
        print(result[1][0])
        print(result[1][1])

    def train_one_epoch(self, epoch):

        num_iteration = min(len(self.sourceloader), len(self.targetloader))
        pbar = tqdm(range(num_iteration), ncols=100, desc=str(epoch))
        for i in pbar:
            p = float(i + epoch * num_iteration) / self.args.epochs / num_iteration
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            simg, slabel = next(iter(self.sourceloader))
            simg, slabel = simg.to(self.device), slabel.to(self.device)

            timg, __ = next(iter(self.targetloader))
            timg = timg.to(self.device)

            ## simply split the model into two???

            #train with source data               
            batch_size = len(slabel)
            domain_label = torch.zeros(batch_size).long().to(self.device)
            self.model._set_alpha = alpha
            if self.cert:
                features = self.model.main_features(simg)
                features = features.view(-1, 50*4*4)
                loss_label, err = robust_loss(self.model.classifier, 0.05, features, slabel)
            else:
                output = self.model(simg)
                loss_label = self.criterion(output, slabel)

            domain_output = self.model(simg, mode='domain')
            loss_domain = self.criterion_domain(domain_output, domain_label)

            # train with target data
            batch_size = len(timg)

            domain_label = torch.ones(batch_size).long().to(self.device)
            domain_output = self.model(timg, mode='domain')
            tloss_domain = self.criterion_domain(domain_output, domain_label)

            loss = loss_label + loss_domain + tloss_domain

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix(loss=loss_label.item(), d_s_loss=loss_domain.item(), d_t_loss=tloss_domain.item())

    def test(self): #source and test
        alpha = 0
        result = []
        for loader in [self.sourcetestloader, self.targettestloader]:
            correct = 0
            length = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = len(images)
                output = self.model(images)
                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                length += batch_size

            accuracy = correct *100//length
            result.append(accuracy)
        return result

    def attack(self):
        RES = []
        for loader in [self.sourcetestloader, self.targettestloader]:
            results = []
            for desc, attack_f in zip(['FGSM', 'PGD'], [self.FGSM, self.PGD]):
                result = []
                for eps in tqdm([i*0.01 for i in range(10)], ncols=100, desc=desc):
                    accuracy = 0
                    length = 0
                    for images, target in loader:
                        images, target = images.cuda(), target.cuda()
                        pert_image = attack_f(eps, images, target)

                        output = self.model(pert_image)
                        pred = output.argmax(dim=1)
                        accuracy += pred.eq(target).data.sum()
                        length += len(pred)
                    result.append(accuracy.item()*100//length)
                results.append(result)
                print(result)

            RES.append(results)
        return RES

    def FGSM(self, eps, images, target):
    ## this is 
        X = images.clone()
        X.requires_grad = True
        output = self.model(X)
        loss = self.criterion(output, target)
        loss.backward()
        grad_sign = X.grad.data.sign()
        return (X + eps*grad_sign).clamp(0, 1)

    def PGD(self, eps, images, target):
        X_orig = images.clone()    
        X_var = images.clone()
        for __ in range(40):
            X = X_var.clone()
            X.requires_grad = True
            output = self.model(X)
            loss = self.criterion(output, target)
            loss.backward()
            grad_sign = X.grad.data.sign()
            X_var = X_var + 0.05*grad_sign
            # X_var.clamp(X_orig-eps, X_orig+eps)
            X_var = torch.where(X_var < X_orig-eps, X_orig-eps, X_var)
            X_var = torch.where(X_var > X_orig+eps, X_orig+eps, X_var)
            X_var.clamp(0, 1)
        return X_var