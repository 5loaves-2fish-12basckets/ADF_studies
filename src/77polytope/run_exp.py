import torch
import json

from module.datafunc import make_dataloaders
from module.models import *
from module.trainer import *
from convex_adversarial import robust_loss

def make_optimizers(model):
    return torch.optim.Adam(model.parameters())

def main():
    trainloader, testloader = make_dataloaders()
    criterion = torch.nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_frames =[
        ('lin',Linear_model)
    ]
    # model_frames =[
    #     ('lin',Linear_model), ('wide', Wide_model), ('deep',Deep_model),
    #     ('vgg',VGG), ('res',ResNet)
    # ]
    # RESULT = {}

    for name, model_func in model_frames:
        model = model_func().cuda()
        robust_model = model_func().cuda()
        optimizer = make_optimizers(model)
        optimizer_r = make_optimizers(robust_model)

        # result = train_test_attack(model, trainloader, testloader, optimizer, criterion)
        # print(result)
        result_c = train_test_attack(robust_model, trainloader, testloader, optimizer_r, criterion, cert=True, robust_loss=robust_loss)
        print(result_c)


        # RESULT[name] = (result, result_cert)
        # model.save('ckpt/'+name+'.pth')
        # robust_model.save('ckpt/'+name+'_cert.pth')


    # with open('ckpt/result.json','w') as f:
    #     json.dump(f, RESULT)

    #plot

if __name__ == '__main__':
    main()