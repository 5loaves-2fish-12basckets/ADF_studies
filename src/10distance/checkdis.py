
import json
import torch
from datafunc import make_dataloaders
from tqdm import tqdm



class config():
    def __init__(self):
        self.data_dir_root = '/home/jimmy/datastore'
        self.img_size = 64
        self.batch_size = 1

config = config()
trainloader, testloader, fontloader = make_dataloaders(config.data_dir_root, config.img_size, config.batch_size, iter_length=1000)
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

# cannot finish 60000 to 60000, random sample 1000 

# fontloader lens is 579

DATA = []
pbar1 = tqdm(enumerate(zip(ids, loaders)), total=len(ids))
for p, (id1, loader1) in pbar1:

    pbar1.set_description('from %s'%names[id1])

    pbar2 = tqdm(zip(ids[p:], loaders[p:]), total=len(ids[p:]))
    for id2, loader2 in pbar2:

        SET = {}
        SET['from'] = names[id1]
        SET['to'] = names[id2]
        SET['samples'] = []

        pbar2.set_description('to %s'%names[id2])
        for i, li in tqdm(loader1):
            for j, lj in tqdm(loader2):
                d0 = l0(i,j)
                d1 = l1(i,j)
                d2 = l2(i,j)
                d8 = l8(i,j)
                sample = {}
                sample['label1']=li.item()
                sample['label2']=lj.item()
                sample['distances']=[d0,d1,d2,d8]
                SET['samples'].append(sample)

        DATA.append(SET)    

with open('out/full_data.json', 'w') as json_file:
    json.dump(DATA, json_file, indent = 4)

## analyse

# avg, min(exclude same), max, between loaders
# avg, min, mac between each labels 



