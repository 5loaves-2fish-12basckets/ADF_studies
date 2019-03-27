# -*- coding: utf-8 -*-
"""Function for data preparation 

This module includes functions used for data preparation. A typical usage would
be to call make_dataset() and make_dataloader() in trainer.py in when training. 

Example:
    dataset = make_dataset(self.config.data_dir_root, self.args)
    dataloader = make_dataloader(dataset, batch_size=self.args.batch_size)

Todo:
    * enlarge list of datasets to include cifar10, lsun, etc

__author__  = '{Jimmy Yeh}'
__email__   = '{marrch30@gmail.com}'
"""

from torch.utils import data
from torchvision import datasets
from torchvision import transforms

def MNIST(data_dir_root, img_size):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset = datasets.MNIST(
        data_dir_root, train=True, download=False, transform=trans
    )
    return dataset


def make_dataloader(data_dir_root, img_size, batch_size):

    trainset = MNIST(data_dir_root, img_size)
    testset = prepare_dataset(img_size)

    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def prepare_dataset(img_size): 

    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    # trainset = datasets.ImageFolder('/home/jimmy/datastore/fonts/', transform=trans)

    testset = datasets.ImageFolder('/home/jimmy/datastore/fonts/digit', transform=trans)

    return testset










'''
previous

import numpy as np
import pandas as pd 
import glob

from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler 

#main function to be called
def make_dataloader(data_dir_root, img_size, batch_size):
    trainset, testset = read_data
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    if datatype=='mnist':
        dataset = MNIST(data_dir_root, img_size)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif datatype=='lsun':
        dataset = LSUN(data_dir_root, img_size)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif datatype=='celeba':
        dataset = CELEBA(data_dir_root, img_size)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def rearrange_data()

def read_data(data_dir_root)
    for file in glob.glob(data_dir_root+'/font/*.csv')
        testdata = []
        traindata = []
        df = pd.read_csv(file)
        for i in len(df):
            values = df.loc[i].values
            # font = values[0]
            ascii_code = values[2]
            # bold_val = values[3] # normal 0 -> bold 1
            # italic = values[4]
            array = values[12:]
            if i%6==0:
                testdata.append((ascii_code, array))
            else:
                traindata.append((ascii_code, array))

    trainset = dataset(traindata)
    testset = dataset

class dataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class Font(data.Dataset):
    def __init__(self, data_dir_root, img_size):
        self.DATA = []


# ['TW.csv', 'MONOTXT.csv', 'BERLIN.csv', 'HAETTENSCHWEILER.csv', 'HIMALAYA.csv', 'PHAGSPA.csv',
#  'NIRMALA.csv', 'BERNARD.csv', 'KRISTEN.csv', 'CENTURY.csv', 'MONOSPAC821.csv', 'VIVALDI.csv',
#   'ARIAL.csv', 'FORTE.csv', 'JUICE.csv', 'HARRINGTON.csv', 'BUXTON.csv', 'ROCKWELL.csv', 'CHILLER.csv',
#    'CURLZ.csv', 'TECHNIC.csv', 'PLAYBILL.csv', 'COUNTRYBLUEPRINT.csv', 'IMPRINT.csv', 'NINA.csv',
#     'HARLOW.csv', 'COOPER.csv', 'EDWARDIAN.csv', 'ONYX.csv', 'LUCIDA.csv', 'GADUGI.csv', 'COMIC.csv',
#      'CAARD.csv', 'BAITI.csv', 'ROMANTIC.csv', 'HANDPRINT.csv', 'FELIX TITLING.csv', 'SKETCHFLOW.csv',
#       'BROADWAY.csv', 'MONEY.csv', 'CASTELLAR.csv', 'GLOUCESTER.csv', 'DUTCH801.csv', 'SNAP.csv', 'VERDANA.csv',
#        'RAVIE.csv', 'TAI.csv', 'CENTAUR.csv', 'WIDE.csv', 'CONSTANTIA.csv', 'VINETA.csv', 'LEELAWADEE.csv',
#         'ELEPHANT.csv', 'COMPLEX.csv', 'CREDITCARD.csv', 'MV_BOLI.csv', 'SHOWCARD.csv', 'TREBUCHET.csv',
#          'PALACE.csv', 'GABRIOLA.csv', 'MODERN.csv', 'GEORGIA.csv', 'BOOK.csv', 'GOUDY.csv', 'NUMERICS.csv',
#           'E13B.csv', 'MISTRAL.csv', 'SWIS721.csv', 'GILL.csv', 'GOTHICE.csv', 'MAGNETO.csv', 'CANDARA.csv',
#            'BITSTREAMVERA.csv', 'RAGE.csv', 'EBRIMA.csv', 'CALIBRI.csv', 'STYLUS.csv', 'PANROMAN.csv',
#             'BASKERVILLE.csv', 'BRITANNIC.csv', 'VINER.csv', 'GARAMOND.csv', 'STENCIL.csv', 'ITALIC.csv',
#              'FRENCH.csv', 'JAVANESE.csv', 'ISOC.csv', 'SERIF.csv', 'COURIER.csv', 'BAUHAUS.csv', 'MAIANDRA.csv',
#               'TAHOMA.csv', 'COPPERPLATE.csv', 'FOOTLIGHT.csv', 'KUNSTLER.csv', 'ROMAN.csv', 'PRISTINA.csv',
#                'SANSSERIF.csv', 'SCRIPTB.csv', 'BRUSH.csv', 'MATURA.csv', 'PERPETUA.csv', 'CITYBLUEPRINT.csv',
#                 'SCRIPT.csv', 'PAPYRUS.csv', 'SYLFAEN.csv', 'QUICKTYPE.csv', 'CONSOLAS.csv', 'CALIFORNIAN.csv',
#                  'GUNPLAY.csv', 'INFORMAL.csv', 'ERAS.csv', 'OCRB.csv', 'MYANMAR.csv', 'RICHARD.csv',
#                   'COMMERCIALSCRIPT.csv', 'PALATINO.csv', 'TEMPUS.csv', 'TXT.csv', 'JOKERMAN.csv', 'SEGOE.csv',
#                    'SIMPLEX.csv', 'ENGRAVERS.csv', 'BANKGOTHIC.csv', 'FREESTYLE.csv', 'PMINGLIU-EXTB.csv',
#                     'VLADIMIR.csv', 'REFERENCE.csv', 'GIGI.csv', 'BODONI.csv', 'BELL.csv', 'OCRA.csv', 'CAMBRIA.csv',
#                      'HIGH TOWER.csv', 'IMPACT.csv', 'VIN.csv', 'AGENCY.csv', 'SITKA.csv', 'BRADLEY.csv', 'PROXY.csv',
#                       'FRANKLIN.csv', 'ENGLISH.csv', 'SUPERFRENCH.csv', 'MINGLIU.csv', 'NIAGARA.csv', 'CALISTO.csv',
#                        'MONOTYPE.csv', 'BLACKADDER.csv', 'CORBEL.csv', 'YI BAITI.csv', 'TIMES.csv', 'BOOKMAN.csv',
#                         'EUROROMAN.csv']






class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


#list of datasets to use
def MNIST(data_dir_root, img_size):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset = datasets.MNIST(
        data_dir_root, train=True, download=True, transform=trans
    )
    return dataset

def LSUN(data_dir_root, img_size):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    classes = ['bedroom_train']
    dataset = datasets.LSUN(
        data_dir_root+'/LSUN',classes=classes, transform=trans
    )
    return dataset


def CELEBA(data_dir_root, img_size):
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = datasets.ImageFolder(
        data_dir_root+'/CELEBA',trans
    )
    return dataset

'''