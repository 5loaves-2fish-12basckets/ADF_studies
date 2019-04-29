
from tqdm import tqdm

def train_test_attack(model, trainloader, testloader, optimizer, criterion, cert=False, robust_loss=None):
    
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
    for eps in tqdm([i*0.01 for i in range(10)], ncols=100):
        accuracy = 0
        length = 0
        pbar = tqdm(testloader, ncols=100)
        pbar.set_description(str(eps))
        for images, target in pbar:
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
    for eps in tqdm([i*0.01 for i in range(10)], ncols=100):
        accuracy = 0
        length = 0
        pbar = tqdm(testloader, ncols=100)
        pbar.set_description(str(eps))
        for images, target in pbar:
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
    X = images.clone()
    X.requires_grad = True
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    grad_sign = X.grad.data.sign()
    return (X + eps*grad_sign).clamp(0, 1)

def PGD(eps, images, target, model, criterion):
    X_orig = images.clone()
    X = images.clone()
    X.requires_grad = True
    output = model(X)
    loss = criterion(output, target)
    loss.backward()
    grad = X.grad.data
    return (X + eps*grad_sign).clamp(0, 1)
