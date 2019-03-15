# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories

from demo_module import inspect

def main():
    config, args, opt = configurations('BASIC_MNIST/attention')
    check_directories(opt.dir_list)
    trainer = Trainer(config, args, opt)
    trainer.load()





if __name__ == '__main__':
    main()

