import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random

def train_test_attack(model, trainloader, testloader, optimizer, criterion, cert=False, robust_loss=None, mode=None):
    
    #train
    
    for epoch in range(1):
        pbar = tqdm(trainloader, ncols=100)
        pbar.set_description(str(epoch))
        for images, target in pbar:
            images, target = images.cuda(), target.cuda()
            output = model(images)
            
            if cert:
                loss, err = robust_loss(model, 0.05, images, target)
            else:
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            accuracy = pred.eq(target).data.sum()*100//len(pred)
            pbar.set_postfix(loss=loss.item(), acc=accuracy.item())

    #test_attack1
    result = []
    for eps in tqdm([i*0.01 for i in range(10)], ncols=100, desc='FGSM'):
        accuracy = 0
        length = 0
        pbar = tqdm(testloader, ncols=100)
        pbar.set_description(str(eps))
        for images, target in pbar:
            if mode is not None:
                images = translation(images, mode)

            images, target = images.cuda(), target.cuda()
            pert_image = FGSM(eps, images, target, model, criterion)

            output = model(pert_image)
            pred = output.argmax(dim=1)
            accuracy += pred.eq(target).data.sum()
            length += len(pred)

        result.append(accuracy.item()*100//length)
    print()
    #test_attack1
    result2 = []
    for eps in tqdm([i*0.01 for i in range(10)], ncols=100, desc='PGD'):
        accuracy = 0
        length = 0
        pbar = tqdm(testloader, ncols=100)
        pbar.set_description(str(eps))
        for images, target in pbar:
            if mode is not None:
                images = translation(images, mode)

            images, target = images.cuda(), target.cuda()
            pert_image = PGD(eps, images, target, model, criterion)

            output = model(pert_image)
            pred = output.argmax(dim=1)
            accuracy += pred.eq(target).data.sum()
            length += len(pred)

        result2.append(accuracy.item()*100//length)
    print()

    return result, result2

def FGSM(eps, images, target, model, criterion):
    ## this is 
    X = images.clone()
    X.requires_grad = True
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    grad_sign = X.grad.data.sign()
    return (X + eps*grad_sign).clamp(0, 1)

def PGD(eps, images, target, model, criterion):
    X_orig = images.clone()    
    X_var = images.clone()
    for __ in range(40):
        X = X_var.clone()
        X.requires_grad = True
        output = model(X)
        loss = criterion(output, target)
        loss.backward()
        grad_sign = X.grad.data.sign()
        X_var = X_var + 0.05*grad_sign
        # X_var.clamp(X_orig-eps, X_orig+eps)
        X_var = torch.where(X_var < X_orig-eps, X_orig-eps, X_var)
        X_var = torch.where(X_var > X_orig+eps, X_orig+eps, X_var)
        X_var.clamp(0, 1)
    return X_var

def translation(images, mode):
    new_images = []
    for image in images:
        image = translate_tri(image, mode)
        new_images.append(image.unsqueeze(0))

    new_images = torch.cat(new_images)
    return new_images


def translate_one(image, mode):
    image = TF.to_pil_image(image)
    if mode == 'rot' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, sign*random.randint(10,30), (0,0), 1, 0)
    if mode == 'trans' or 'all':
        width, height = image.size
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltaw = sign*random.randint(int(0.1*width), int(0.2*width))
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltah = sign*random.randint(int(0.1*height), int(0.2*height))
        image = TF.affine(image, 0, (deltaw, deltah), 1, 0)
    if mode == 'scale' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, 0, (0,0), 1 + sign * 0.1, 0)

    image = TF.to_tensor(image)
    return image

def translate_two(image, mode):
    image = TF.to_pil_image(image)
    if mode == 'rot' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, sign*random.randint(1,5), (0,0), 1, 0)
    if mode == 'trans' or 'all':
        width, height = image.size
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltaw = sign*random.randint(int(0.01*width), int(0.02*width))
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltah = sign*random.randint(int(0.01*height), int(0.02*height))
        image = TF.affine(image, 0, (deltaw, deltah), 1, 0)
    if mode == 'scale' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, 0, (0,0), 1 + sign * 0.01, 0)

    image = TF.to_tensor(image)
    return image

def translate_tri(image, mode):
    image = TF.to_pil_image(image)
    if mode == 'rot' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, sign*random.randint(5,10), (0,0), 1, 0)
    if mode == 'trans' or 'all':
        width, height = image.size
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltaw = sign*random.randint(int(0.05*width), int(0.1*width))
        sign = (int(random.random() < 0.5) - 0.5) * 2
        deltah = sign*random.randint(int(0.05*height), int(0.1*height))
        image = TF.affine(image, 0, (deltaw, deltah), 1, 0)
    if mode == 'scale' or 'all':
        sign = (int(random.random() < 0.5) - 0.5) * 2
        image = TF.affine(image, 0, (0,0), 1 + sign * 0.05, 0)

    image = TF.to_tensor(image)
    return image

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from module.datafunc import make_dataloaders
    from torchvision.utils import save_image
    trainloader, testloader = make_dataloaders(batch_size=1)
    for image, __ in testloader:
        image1 = translate_one(image.squeeze(0), mode='all')   
        save_image(image1, 'img1.png')
        image2 = translate_two(image.squeeze(0), mode='all')   
        save_image(image2, 'img2.png')
        image3 = translate_tri(image.squeeze(0), mode='all')   
        save_image(image3, 'img3.png')
        break
