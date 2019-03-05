# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

import sys
sys.path.append('..')
sys.path.append('.')

from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories

def main():
    config, args, opt = configurations('BASIC_MDWNET')
    ## simply run a legit mdwnet
    trainer = Trainer(config, args, opt)
    trainer.train()

if __name__ == '__main__':
    main()

