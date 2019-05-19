# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from dann_module.addamodel2 import ADDA 
from dann_module.datafunc import make_dataloaders
from tqdm import tqdm
import torch
import os

from math import sqrt

torch.manual_seed(7)
torch.cuda.manual_seed_all(100)
torch.backends.cudnn.deterministic = True 

class Trainer():
    def __init__(self, args):
        self.model = ADDA()
        self.optimizer = torch.optim.Adam(
                list(self.model.encoder.parameters()) \
                + list(self.model.classifier.parameters()),
                lr=1e-3, betas=(0.5, 0.9)
            )
        self.ten_optimizer = torch.optim.Adam(self.model.tencoder.parameters(),
            lr=1e-4, betas=(0.5, 0.9))
        self.tel_optimizer = torch.optim.Adam(self.model.teller.parameters(),
            lr=1e-4, betas=(0.5, 0.9))
        
        self.criterion = torch.nn.CrossEntropyLoss()

        dataloaders = make_dataloaders(args.source, args.target, args.batch_size)
        self.sourceloader, self.targetloader, self.testtargetloader = dataloaders

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.args = args
        self.modelpath = os.path.join('ckpt', args.taskname, 'model_%s.pth'%(args.target[:2]))
        self.bestpath = os.path.join('ckpt', args.taskname, 'best_%s.pth'%(args.target[:2]))
        self.pretrain_path = os.path.join('ckpt', args.taskname, 'pre_%s.pth'%(args.source[:2]))
        self.tarstep = 1

    def train(self):
        print('%s --> %s'%(self.args.source, self.args.target))

        if os.path.exists(self.pretrain_path):
            self.model.load_pretrain(self.pretrain_path)        
        else:
            self.pretrain()
        self.model.target_load_source()
        self.adapt_target()
        print()

        
    def pretrain(self):
        self.model.encoder.train()
        self.model.classifier.train()
        # bbar = tqdm(range(100), ncols=100, desc='pretrain')
        bbar = tqdm(range(self.args.epochs), ncols=100, desc='pretrain')
        for epoch in bbar:
            pbar = tqdm(self.sourceloader, ncols=100, desc='tr '+str(epoch))
            accuracy = 0
            length = 0

            for img, label in pbar:
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, label)

                loss.backward()
                self.optimizer.step()
                acc = output.argmax(1).eq(label).sum()*100//len(label)
                accuracy += output.argmax(1).eq(label).sum().item()
                length += len(label)
                pbar.set_postfix(loss=loss.item(), acc=acc.item())
            accuracy = accuracy * 100 / length
            bbar.set_postfix(acc=accuracy)
        self.model.save(self.pretrain_path)

    def adapt_target(self):
        self.model.tencoder.train()
        self.model.teller.train()

        num_iteration = min(len(self.sourceloader), len(self.targetloader))
        bbar = tqdm(range(500), ncols=100, desc='adapt')
        # bbar = tqdm(range(self.args.epochs), ncols=100, desc='adapt')
        best_acc = 0

        for epoch in bbar:
            total_acc=0
            length=0
            target_acc=0
            tlength=0
            pbar = tqdm(range(num_iteration), ncols=100, desc='ada '+str(epoch))
            for i in pbar:
                simg, __ = next(iter(self.sourceloader))
                timg, __ = next(iter(self.targetloader))
                simg, timg = simg.to(self.device), timg.to(self.device)
                batch_size = len(simg)

                # train teller
                self.tel_optimizer.zero_grad()
                source_feature = self.model.encoder(simg)
                target_feature = self.model.tencoder(timg)
                concat_feature = torch.cat((source_feature, target_feature), 0)

                concat_dom = self.model.teller(concat_feature.detach())

                source_lab = torch.ones(batch_size).long().to(self.device)
                target_lab = torch.zeros(batch_size).long().to(self.device)
                concat_lab = torch.cat((source_lab, target_lab), 0)

                tell_loss = self.criterion(concat_dom, concat_lab)
                tell_loss.backward()

                self.tel_optimizer.step()
                afloat = concat_dom.argmax(1).eq(concat_lab).float()
                acc = afloat.mean().item()
                total_acc += afloat.sum().item()
                length += len(afloat)

                ## train tencoder
                for __ in range(self.tarstep):
                    self.ten_optimizer.zero_grad()
                    self.tel_optimizer.zero_grad()

                    target_feature = self.model.tencoder(timg)
                    target_dom = self.model.teller(target_feature)

                    target_lab = torch.ones(batch_size).long().to(self.device)

                    targ_loss = self.criterion(target_dom, target_lab)
                    targ_loss.backward()
                    self.ten_optimizer.step()

                    bfloat = target_dom.argmax(1).eq(target_lab).float()
                    tacc = bfloat.mean().item()
                    target_acc+= bfloat.sum().item()
                    tlength+= len(bfloat)
                    pbar.set_postfix(teller=acc, target=tacc)


                if 0.3 < tacc < 0.4:
                    self.tarstep = 2
                elif 0.2 < tacc < 0.3:
                    self.tarstep = 3
                elif 0.15 < tacc < 0.2:
                    self.tarstep = 4
                elif 0.1 < tacc < 0.15:
                    self.tarstep = 5
                elif tacc < 0.1:
                    self.tarstep = 6
                else:
                    self.tarstep = 1

            total_acc = total_acc * 100 / length  ## total domain accuracy
            target_acc = target_acc * 100 / tlength
            acc, loss = self.test()
            if acc>best_acc:
                best_acc=acc
                self.model.save(self.bestpath)
            self.model.save(self.modelpath)
            bbar.set_postfix(acc=acc, best_acc=best_acc, tar=target_acc, tel=total_acc)

    def test(self):
        self.model.eval()
        loss = 0 
        acc = 0
        length = 0
        pbar = tqdm(self.testtargetloader, ncols=100, desc='test')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images, mode='target')
            loss += self.criterion(output, labels).item()

            pred = output.argmax(1)
            acc += pred.eq(labels).sum().item()
            length += len(labels)

        loss /= length
        acc /= length
        return acc, loss



