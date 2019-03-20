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
    result = []
    for i in range(1,10):
        opt.epochs = i
        trainer = Trainer(config, args, opt)
        result.append(trainer.train())

    print(result)

if __name__ == '__main__':
    main()