# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from dann_module.addamodel import ADDA 
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
            )
        self.ten_optimizer = torch.optim.Adam(self.model.tencoder.parameters())
        self.tel_optimizer = torch.optim.Adam(self.model.teller.parameters())
        
        self.criterion = torch.nn.CrossEntropyLoss()

        dataloaders = make_dataloaders(args.source, args.target, args.batch_size)
        self.sourceloader, self.targetloader, self.testtargetloader = dataloaders

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.args = args
        self.modelpath = os.path.join('ckpt', args.taskname, 'model_%s_%s.pth'%(args.source[:1], args.target[:1]))
        self.bestpath = os.path.join('ckpt', args.taskname, 'best_%s_%s.pth'%(args.source[:1], args.target[:1]))
        self.dstep = 1
        self.tstep = 1

    def train(self):
        print('%s --> %s'%(self.args.source, self.args.target))
        best_acc = 0
        for i in range(1):
            self.train_one_epoch(i)
        for i in range(1):
            self.adjust_one_epoch(i)

        acc, loss = self.test()
        print(acc)


        # bbar = tqdm(range(self.args.epochs), ncols=100)
        # for epoch in bbar:
        #     train_acc, train_loss = self.train_one_epoch(epoch)
        #     dom_acc, dom_loss, tar_acc, tar_loss = self.adjust_one_epoch(epoch)
        #     acc, test_loss = self.test()

        #     self.model.save(self.modelpath)
        #     if acc > best_acc:
        #         best_acc = acc
        #         torch.save(self.model.state_dict(), self.bestpath)
        #     bbar.set_postfix(ac=acc, bacc=best_acc, dac=dom_acc, tar=tar_acc, tracc=train_acc)

    def train_one_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.sourceloader, ncols=100, desc='tr'+str(epoch))
        
        for img, label in pbar:
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.criterion(output, label)

            loss.backward()
            self.optimizer.step()
            acc = output.argmax(1).eq(label).sum()*100//len(label)
            pbar.set_postfix(loss=loss.item(), acc=acc.item())

        self.model.copy_target()
        return acc.item(), loss.item()

    def adjustpq(self, dom_acc, tar_acc):
        self.dstep = sqrt((1-dom_acc)*5) + 1
        if tar_acc > 0.8:
            self.tstep -= 0.1
            self.tstep = max(1, self.tstep)
        if tar_acc > 0.9:
            self.tstep -= 0.1
            self.tstep = max(1, self.tstep)

        if tar_acc < 0.5:
            self.tstep += 0.1
        else:
            self.tstep = min(3, self.tstep)
        if tar_acc < 0.4:
            self.tstep += 0.1
        else:
            self.tstep = min(3, self.tstep)
        if tar_acc < 0.3:
            self.tstep += 0.2
        else:
            self.tstep = min(4, self.tstep)
        if tar_acc < 0.2:
            self.tstep += 0.3
        else:
            self.tstep = min(6, self.tstep)
        if tar_acc < 0.1:
            self.tstep += 0.4
        else:
            self.tstep = min(8, self.tstep)
        if tar_acc < 0.05:
            self.tstep += 0.5
        else:
            self.tstep = min(11, self.tstep)

    def adjust_one_epoch(self, epoch):
        self.model.train()
        num_iteration = min(len(self.sourceloader), len(self.targetloader))
        pbar = tqdm(range(num_iteration), ncols=100, desc='adj'+str(epoch))

        for i in pbar:
            for __ in range(int(self.dstep)):
                timg, __ = next(iter(self.targetloader))
                simg, __ = next(iter(self.sourceloader))
                timg, simg = timg.to(self.device), simg.to(self.device)
                batch_size = len(timg)

                # train teller
                self.tel_optimizer.zero_grad()
                source_dom = self.model(simg, mode='domains')
                target_dom = self.model(timg, mode='domain')
                
                domain_label = torch.ones(batch_size).long().to(self.device)
                src_dom_loss = self.criterion(source_dom, domain_label)
                src_acc = source_dom.argmax(1).eq(domain_label).float().mean().item()

                domain_label = torch.zeros(batch_size).long().to(self.device)
                tar_dom_loss = self.criterion(target_dom, domain_label)
                tar_acc = target_dom.argmax(1).eq(domain_label).float().mean().item()

                dom_loss = (src_dom_loss + tar_dom_loss)/2

                dom_loss.backward()
                self.tel_optimizer.step()

                dom_acc = (src_acc + tar_acc)/2


            ## train tencoder
            for __ in range(int(self.tstep)):
                self.ten_optimizer.zero_grad()
                self.tel_optimizer.zero_grad()

                dom_output = self.model(timg, mode='domain')
                domain_label = torch.ones(batch_size).long().to(self.device)

                tar_loss = self.criterion(dom_output, domain_label)
                tar_loss.backward()
                tar_acc = dom_output.argmax(1).eq(domain_label).float().mean().item()
                self.ten_optimizer.step()

            # self.adjustpq(dom_acc, tar_acc)
            pbar.set_postfix(dom_acc=dom_acc, tar_acc=tar_acc, dstep=self.dstep, tstep=self.tstep)


        return dom_acc, dom_loss.item(), tar_acc, tar_loss.item()

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



