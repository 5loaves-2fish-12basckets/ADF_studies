import torch
import json

from module.datafunc import make_dataloaders
from module.models import *
from module.trainer import *
from convex_adversarial import robust_loss, robust_loss_parallel

def make_optimizers(model):
    return torch.optim.Adam(model.parameters())

def main():
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_frames =[
    #     ('lin',Linear_model),
    # ]
    model_frames =[
        ('lin',Linear_model), 
        ('wide', Wide_model), 
        ('deep',Deep_model),
        # ('vgg',VGG), 
        # ('res',ResNet)
    ]
    RESULT = {}

    for name, model_func in model_frames:
        if name=='wide':
            batch_size = 16
        elif name=='deep':
            batch_size = 8
        else:
            batch_size = 128
        trainloader, testloader = make_dataloaders(batch_size=batch_size)

        print(name)
        model = model_func().cuda()
        optimizer = make_optimizers(model)
        result = train_test_attack(model, trainloader, testloader, optimizer, criterion)
        print(result)
        
        torch.save(model.state_dict(), 'ckpt/'+name+'.pth')
        del model, optimizer
        torch.cuda.empty_cache()

        ## robust        
        robust_model = model_func().cuda()
        optimizer_r = make_optimizers(robust_model)
        result_c = train_test_attack(robust_model, trainloader, testloader, optimizer_r, criterion, cert=True, robust_loss=robust_loss)
        print(result_c)

        torch.save(robust_model.state_dict(), 'ckpt/'+name+'_cert.pth')
        del robust_model, optimizer_r
        torch.cuda.empty_cache()

        RESULT[name] = (result, result_c)

    with open('ckpt/result.json','w') as f:
        json.dump(RESULT, f)

    #plot

if __name__ == '__main__':
    main()

'''
model.load_state_dict(torch.load('ckpt/'+name+'.pth'))
result_c = train_test_attack(robust_model, trainloader, testloader, optimizer_r, criterion, cert=True, robust_loss=robust_loss_parallel)

'''