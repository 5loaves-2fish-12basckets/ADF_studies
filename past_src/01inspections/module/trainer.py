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

import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import glob

from PIL import Image
from shutil import copyfile


class Trainer():
    def __init__(self):
        self.models = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.load_models()

        self.img_size = 64
        self.trans = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
                ])
    def load_models(self):
        print('loading models')
        for modelpath in tqdm(glob.glob('output/vgg/*-8.pth')):
            model = VGG()
            model.to(self.device)
            model.load_state_dict(torch.load(modelpath))
            self.models.append(model)

    def inspect_data(self):# and copy to dest
        ge = []
        for i in range(10):
            count = 0
            for file in tqdm(glob.glob('/home/jimmy/datastore/fonts/digit/%d/*.png'%i)):
                img = Image.open(file)
                img_tensor = self.trans(img).to(self.device).unsqueeze(0)
                corr = 0
                total = 0
                for vggmodel in self.models:
                    output = vggmodel(img_tensor)
                    pred = output.argmax(dim=1, keepdim=True)
                    if pred == i:
                        corr +=1
                    total+=1
                # print(file, 'corr %d/ total %d is %d%%'%(corr, total, corr*100//total))
                if corr*100//total>95:
                    count+=1
                    dest = file.split('/')[-1]
                    copyfile(file, 'select_data/%d/%s'%(i,dest))
            ge.append(count)
        for i in range(10):
            print('num %d total %d >95%%'%(i, ge[i]))




