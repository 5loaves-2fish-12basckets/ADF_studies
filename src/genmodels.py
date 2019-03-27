from module.trainer import Trainer
from module.config import configurations
from module.utils import check_directories,  gpu_gauge

import torch.multiprocessing as mp
import numpy as np

from tqdm import tqdm
import time
def gen_model(mtype, rd, ep):
    config, args, opt = configurations('CHOOSE')

    config.modeltype = mtype
    config.random = rd
    opt.epochs = ep
    trainer = Trainer(config, args, opt)
    trainer.train()

def test_model(fpath, spath, rd, mtype):
    config, args, opt = configurations('CHOOSE')
    config.resume = True
    config.resume_path = fpath
    config.random = rd
    config.modeltype = mtype
    trainer = Trainer(config, args, opt)
    acc, corr_array = trainer.test()

    # np.save(spath, (acc,corr_array))
    np.save(spath, corr_array)

def main():
    memory = gpu_gauge()
    processes = []
    workinprogress = []
    workinprogress_test = []
    processes_test = []

    # for mtype in ['res', 'vgg']:
    #     for rd in range(1):
    #         for ep in [1]:
    #             processes.append(mp.Process(target=gen_model,\
    #                 args = (mtype, rd, ep)))

#### generate models
    # for mtype in ['res', 'vgg']:
    #     for rd in range(100):
    #         for ep in [10]:
    #             processes.append(mp.Process(target=gen_model,\
    #                 args = (mtype, rd, ep)))
    # n_remain = len(processes)
    # print('total of gen model processes:',n_remain)
    # for process in processes:
    #     while memory.available() < 4000:
    #         time.sleep(10)
    #     process.start()
    #     process.join(0.1)
    #     workinprogress.append(process)
    #     time.sleep(5)
    #     n_remain-=1
    #     print('start one process, %d gen model remain'%n_remain)

    # for process in tqdm(workinprogress):
    #     while process.is_alive():
    #         time.sleep(2)


#### test models

    # for mtype in ['res', 'vgg']:
    #     for rd in range(1):
    #         for ep in range(1):
    #             fpath = 'output/%s/%d-%d.pth'%(mtype, rd, ep)
    #             spath = 'output/%s/%d-%d.npy'%(mtype, rd, ep)
    #             processes_test.append(mp.Process(target=test_model,\
    #                 args = (fpath, spath, rd, mtype)))
    for mtype in ['vgg']:
    # for mtype in ['res', 'vgg']:
        for rd in range(100):
            for ep in range(10):
                fpath = 'output/%s/%d-%d.pth'%(mtype, rd, ep)
                spath = 'output/%s/%d-%d.npy'%(mtype, rd, ep)
                processes_test.append(mp.Process(target=test_model,\
                    args = (fpath, spath, rd, mtype)))

    n_remain = len(processes_test)
    print('total of test model processes:',n_remain)
    n_count = 0
    for process in processes_test:
        while memory.available() < 4000:
            time.sleep(2)
        process.start()
        process.join(0.1)
        workinprogress_test.append(process)
        time.sleep(0.6)
        # time.sleep(30)
        n_remain-=1
        print('start one process, %d test model remain'%n_remain)
        n_count +=1
        if n_count%7 ==0:
            time.sleep(2)

        if n_count > 100:
            for process in tqdm(workinprogress_test):
                while process.is_alive():
                    time.sleep(0.1)
            workinprogress_test = []
            n_count =0

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()