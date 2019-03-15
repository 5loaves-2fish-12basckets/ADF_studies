# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.model import LeNet5
from module.datafunc import make_dataloader, MNIST
from module.utils import check_directories

from grad_cam import GradCAM, BackPropagation

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

from tqdm import tqdm
import numpy as np
import cv2
import glob

class Trainer():
    def __init__(self, config, args, opt):
        self.model = LeNet5(args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.args = args
        self.opt = opt

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device =='cuda':
            self.model.cuda()

    def main_inspection(self):
        # images = [cv2.imread(file) for file in glob.glob(self.opt.task_dir+'/adsamples/*.png')]

        # for raw_image in images:
        # # raw_image = cv2.imread(self.opt.task_dir+'/adsamples')
        # # raw_image = cv2.imread(image_path)[..., ::-1]
        #     image = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         ]
        #     )(raw_image).unsqueeze(0)
        #     image = image.to(self.device)
        #     self.inspect_one(image)
        for i,img in enumerate(self.origsamples):
            self.inspect_one(img, 'o'+str(i))
        for i,img in enumerate(self.adsamples):
            # save_image(img, 'myresults/test.png')
            self.inspect_one(img, 'a'+str(i))

    def inspect_one(self, image, name):
        def save_gradcam(filename, gcam, raw_image):
            # print(raw_image.shape)
            h, w, __= raw_image.shape
            gcam = cv2.resize(gcam, (w, h))
            gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
            gcam = gcam.astype(np.float)/100 + raw_image.astype(np.float)
            gcam = gcam / gcam.max() * 255.0
            cv2.imwrite(filename, np.uint8(gcam))

        # print("Grad-CAM")

        bp = BackPropagation(model=self.model)
        predictions = bp.forward(image)
        print(predictions)
        bp.remove_hook()

        gcam = GradCAM(model=self.model)
        _ = gcam.forward(image)

        classes = list(range(10))
        # print(self.model)
        for i in range(1):
            # print("[{:.5f}] {}".format(predictions[i][0], classes[predictions[i][1]]))

            # Grad-CAM
            gcam.backward(idx=predictions[i][1])

            region = gcam.generate(target_layer='conv')
            
            img = image.squeeze().detach().cpu().numpy()
            rgbimg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            save_gradcam(
                "myresults/{}-{}-{}.png".format(
                    name, 'conv', classes[predictions[i][1]]
                ),
                region,
                rgbimg,
            )


    def load(self):
        self.model.load(self.opt.model_filepath)

    def adversary(self):
        print('adversary for task:', self.config.taskname)
        adsample_path = self.opt.task_dir+'/adsamples'
        check_directories([adsample_path])

        self.adsamples = []
        self.origsamples = []
        test_set = MNIST(self.config.data_dir_root, self.args.img_size, False)
        epsilon = 0.3

        count = 1
        list_ = []
        for i in range(len(test_set)):
            image, target = test_set[i]
            image = image.to(self.device)
            image = image.unsqueeze(0)
            target = torch.LongTensor([target]).to(self.device)
            output = self.model(image)
            if not output.argmax(dim=1).eq(target):
                continue
            elif target in list_:
                continue

            self.origsamples.append(image)
            list_.append(target)

            image.requires_grad = True
            output = self.model(image)
            loss = self.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            gradient = image.grad.data
            adimg = self.FGSM(image, epsilon, gradient)
            self.adsamples.append(adimg)
            pred = self.model(adimg).argmax(dim=1).item()

            save_image(adimg.cpu(), adsample_path+'/sample%d_%d-%d.png'%(count, target.item(), pred))
            
            count+=1
            if count > 10:
                break

    def FGSM(self,img, eps, grad):
        adimg = img + eps*grad.sign()
        adimg = torch.clamp(adimg, 0, 1)
        return adimg

    def train(self):
        print('training for task:', self.config.taskname)
        train_loader, test_loader = make_dataloader(   self.config.data_dir_root, 
                                        self.config.datatype, 
                                        self.args.img_size,
                                        self.args.batch_size)



        print('train %d #'%self.opt.epochs, 'save at %s'%self.opt.task_dir)

        for i in range(self.opt.epochs):
            self.train_one_epoch(train_loader)
        self.test(test_loader)
        self.model.save(self.opt.model_filepath)

    def train_one_epoch(self, train_loader):
        pbar = tqdm(train_loader)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            output = self.model(inputs)
            loss = self.criterion(output, targets)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred =  output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(targets.view_as(pred)).sum().item()
            accuracy = accuracy/len(targets) * 100

            message = 'loss: %.4f, accuracy: %.2f %%'%(loss.item(), accuracy)
            pbar.set_description(message)


    def test(self, test_loader):
        pbar = tqdm(test_loader)
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

