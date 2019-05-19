
import torch
import torchvision

from PIL import Image
import os
import csv


# data_root = 'hw3_data/digits'
# dir_list = ['mnistm', 'svhm', 'usps']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root='hw3_data/digits', data_name='mnistm', transform=None, train=True):
        
        folder = 'train' if train else 'test'
        self.dir = os.path.join(data_root, data_name, folder)
        self.labelpath = os.path.join(data_root, data_name, folder+'.csv')

        example_filename = os.listdir(self.dir)[0].split('/')[-1].split('.')[0]

        self.k = len(str(example_filename))
        self.length = len(os.listdir(self.dir))
        self.str_ = lambda i: '0' * (self.k - len(str(i))) + str(i)

        self._readlabel()
        self.trans = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        imgfile = '%s.png'%self.str_(index)
        label = self.labeldict[imgfile]
        label = torch.LongTensor([label]).squeeze()
        
        imgpath = os.path.join(self.dir, imgfile)
        img = Image.open(imgpath)
        img = self.trans(img)
        img = img.expand(3, 28, 28)  
        return img, label

    def _readlabel(self):
        self.labeldict = {}
        with open(self.labelpath, newline='') as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if first:
                    first = False
                else:
                    self.labeldict[row[0]]=int(row[1])

def make_dataloaders(source, target, batch_size):
    sourceset = Dataset(data_name=source, train=True)
    sourcetestset = Dataset(data_name=source, train=False)
    targetset = Dataset(data_name=target, train=True)
    targettestset = Dataset(data_name=target, train=False)

    sourceloader = torch.utils.data.DataLoader(sourceset, batch_size=batch_size, shuffle=True)
    sourcetestloader = torch.utils.data.DataLoader(sourcetestset, batch_size=batch_size, shuffle=False)
    targetloader = torch.utils.data.DataLoader(targetset, batch_size=batch_size, shuffle=True)
    targettestloader = torch.utils.data.DataLoader(targettestset, batch_size=batch_size, shuffle=False)

    return sourceloader, sourcetestloader, targetloader, targettestloader

def make_loader(source):
    dset = Dataset(data_name=source, train=False)
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)
    return loader

def mnist_dataloaders(batch_size):
    trans = torchvision.transforms.ToTensor()
    sourceset = torchvision.datasets.MNIST('/home/jimmy/dataset/mnist', train=True, download=False, transform=trans)
    targetset = Dataset(data_name='mnistm', train=True)
    testtargetset = Dataset(data_name='mnistm', train=False)

    sourceloader = torch.utils.data.DataLoader(sourceset, batch_size=batch_size, shuffle=True)
    targetloader = torch.utils.data.DataLoader(targetset, batch_size=batch_size, shuffle=True)
    testtargetloader = torch.utils.data.DataLoader(testtargetset, batch_size=batch_size, shuffle=False)

    return sourceloader, targetloader, testtargetloader

class Singleset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.filelist = os.listdir(data_dir)
        self.length = len(self.filelist)
        self.trans = torchvision.transforms.ToTensor()
        self.data_dir = data_dir
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        imgname = self.filelist[index]
        imgpath = os.path.join(self.data_dir, imgname)
        img = Image.open(imgpath)
        img = self.trans(img)

        return img, imgname

def make_a_dataloader(data_dir):
    singleset = Singleset(data_dir)
    singleloader = torch.utils.data.DataLoader(singleset, batch_size=128, shuffle=False)
    return singleloader


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    domain_list = ['usps', 'mnistm', 'svhn', 'usps']
    for i in range(3):
        source = domain_list[i]
        target = domain_list[i+1]
        print(source, target)
        
        loader1, loader2, loader3 = make_dataloaders(source, target, 128)
        for loader in [loader1, loader2, loader3]:
            for image, label in loader:
                print(image.shape)
                print(label.shape)
                print(label)
                break

    print('-====-====-====-')
    loader = make_a_dataloader('hw3_data/digits/mnistm/test')
    for image, name in loader:
        print(name)
        print(image.shape)
        break

## usps 1 --> 3