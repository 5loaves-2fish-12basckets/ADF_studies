from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories,  gpu_gauge

import torch.multiprocessing as mp
import numpy as np

import time

def single_task(mtype, rd, ep):
    config, args, opt = configurations('CHOOSE')
    check_directories(opt.dir_list)
    result = []

    config.modeltype = mtype
    config.random = rd
    opt.epochs = ep
    trainer = Trainer(config, args, opt)
    result = trainer.train()
    np.save('output/CHOOSE/%s%d.npy'%(mtype, rd), result)


def main():
    memory = gpu_gauge()
    processes = []
    # for mtype in ['vgg']:
    #     for rd in range(1):
    #         for ep in [1]:
    #             processes.append(mp.Process(target=single_task,\
    #                 args = (mtype, rd, ep)))
    for mtype in ['res', 'vgg']:
        for rd in range(100):
            for ep in [10]:
                processes.append(mp.Process(target=single_task,\
                    args = (mtype, rd, ep)))
    n_remain = len(processes)
    print('total of processes:',n_remain)
    for process in processes:
        while memory.available() < 4000:
            time.sleep(2)
        process.start()
        process.join(0.1)
        time.sleep(5)
        n_remain-=1
        print('start one process, %d remain'%n_remain)

### unfinished

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()




