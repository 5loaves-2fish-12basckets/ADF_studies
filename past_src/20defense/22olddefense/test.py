import torch
import torchvision
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack

from module.datafunc import make_dataloader
from module.model import VGG
from tqdm import tqdm
import numpy as np

# model = torchvision.models.vgg16(pretrained=False)


def main():

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    trainloader, __ = make_dataloader('/home/jimmy/datastore', 64, 128)

    testset = torchvision.datasets.MNIST('/home/jimmy/datastore', train=False, download=False, transform=trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    testoneloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    model = VGG()
    model = model.cuda()
    adversary = FGSMAttack(model, epsilon=0.5)
    advset = []
    for (X,y) in tqdm(testoneloader, ncols=100):
        adX = adversary.perturb(X, y)
        adset = (torch.from_numpy(adX).squeeze(0),y.squeeze())
        advset.append(adset)

    advloader = torch.utils.data.DataLoader(advset, batch_size=128, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(model, trainloader, optimizer, criterion,)
    test(model, testloader)
    test(model, advloader, 'adversarial attack')

    model = VGG()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    adversary = FGSMAttack(model, epsilon=0.5)
    train(model, trainloader, optimizer, criterion, adv=True, adversary=adversary)
    test(model, testloader)
    test(model, advloader, 'adversarial attack')


def train(model, loader, optimizer, criterion, epochs=1, adv=False, adversary=None):
    for i in range(epochs):
        pbar = tqdm(loader, ncols=100)
        pbar.set_description('train: '+str(i))
        for image, target in pbar:
            image, target = image.cuda(), target.cuda()
            pred = model(image)
            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(target.view_as(pred)).sum().item()*100//len(pred)
            if adv:
                adimage = adversary.perturb(image.cpu(), target.cpu())
                adimage = torch.from_numpy(adimage).cuda()
                pred2 = model(adimage)
                optimizer.zero_grad()
                loss = criterion(pred2, target)
                loss.backward()
                optimizer.step()
                pred2 = pred2.argmax(dim=1, keepdim=True)
                accuracy2 = pred2.eq(target.view_as(pred2)).sum().item()*100//len(pred2)
                pbar.set_postfix(loss=loss.item(), acc=accuracy, adacc=accuracy2)
            else:
                pbar.set_postfix(loss=loss.item(), acc=accuracy)


## try attack
# adversaries = [FGSMAttack(model, epsilon=i) for i in [0.1,0.2,0.3,0.4,0.5]]

def test(model, dataloader, name='test'):
    accuracy = 0
    length=0
    for image, target in tqdm(dataloader, ncols=100):
        image, target = image.cuda(), target.cuda()
        pred = model(image)
        pred = pred.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        length += len(pred)

    print(name + ' result:')
    print(accuracy*100//length)

## try adversarial training
if __name__ == '__main__':
    main()