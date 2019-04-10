# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories

def main():
    config, args, opt = configurations('BASIC')
    check_directories(opt.dir_list)
    config.modeltype = 'vgg'
    config.random = 1
    opt.epochs = 5
    trainer = Trainer(config, args, opt)
    trainer.train()
    print(trainer.test())

if __name__ == '__main__':
    main()