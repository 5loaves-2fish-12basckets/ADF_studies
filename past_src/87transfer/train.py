# -*- coding: utf-8 -*-
"""
__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""
import os
import sys
sys.path.append('.')
import argparse

from module.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', type=str, default='transfer', help='taskname for model saving etc')
    parser.add_argument('--resume', type=str, default=None, help='resume model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--cert', type=bool)

    args = parser.parse_args()
    print('taskname', args.taskname, '# epochs', args.epochs, 'batch_size', args.batch_size,)
    
    if not os.path.exists('ckpt/%s'%args.taskname):
        os.mkdir('ckpt/%s'%args.taskname)
        print('created', 'ckpt/%s'%args.taskname)

    domain_list = ['usps', 'mnistm']
    # domain_list = ['mnistm', 'svhn', 'usps']
    # domain_list = ['usps', 'mnistm', 'svhn', 'usps']
    for cert in [True, False]:
        args.cert = cert
        print('cert', cert)
        for i in range(len(domain_list)-1):
            source = domain_list[i]
            target = domain_list[i+1]
            args.source = source
            args.target = target

            trainer = Trainer(args)
            trainer.train()
            print()

if __name__ == '__main__':
    main()
