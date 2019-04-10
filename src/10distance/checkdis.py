import numpy as np
import torch
from datafunc import make_dataloaders
from tqdm import tqdm
class config():
    def __init__(self):
        self.data_dir_root = '/home/jimmy/datastore'
        self.img_size = 64
        self.batch_size = 1

config = config()
trainloader, testloader, fontloader = make_dataloaders(config.data_dir_root, config.img_size, config.batch_size)
ids = [1,2,3] # mnist_train=1 , mnist_test=2, font_digit_select = 3
loaders = [trainloader, testloader, fontloader]

names = {1:'mnist_train',2:'mnist_test', 3:'font_digit_select'}
## distance metrics
def l0(i,j):
    return torch.dist(i,j,0).item()
def l1(i,j):
    return torch.dist(i,j,1).item()
def l2(i,j):
    return torch.dist(i,j,2).item()
def l8(i,j):
    return torch.dist(i,j,float('inf')).item()

## create distance tables
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
DATA = []
pbar1 = tqdm(zip(ids, loaders), total=len(ids))
for id1, loader1 in pbar1:
    pbar1.set_description('from %s'%names[id1])
    pbar2 = tqdm(zip(ids, loaders), total=len(ids))
    for id2, loader2 in pbar2:
        pbar2.set_description('to %s'%names[id2])
        for i, li in tqdm(loader1):
            for j, lj in tqdm(loader2):
                i = i
                j = j
                d0 = l0(i,j)
                d1 = l1(i,j)
                d2 = l2(i,j)
                d8 = l8(i,j)
                DATA.append([id1, li.item(), id2, lj.item(), d0, d1, d2, d8])

DATA = np.array(DATA)
np.save('full_data.npy', DATA)
## analyse

# avg, min(exclude same), max, between loaders
# avg, min, mac between each labels 



