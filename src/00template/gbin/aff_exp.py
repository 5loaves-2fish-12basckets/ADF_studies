import torch
import json

# from module.datafunc import make_dataloaders
# from module.models import *
# from module.trainer import *
from convex_adversarial import robust_loss, robust_loss_parallel

from dannmodule.model import DaNN
from dannmodule.datafunc import make_dataloaders


def make_optimizers(model):
    return torch.optim.Adam(model.parameters())

def main():
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model_frames =[
    #     ('lin',Linear_model),
    # ]
    model_frames =[
        ('transfer', DaNN)
        # ('lin',Linear_model), 
        # ('wide', Wide_model), 
        # ('deep',Deep_model),

        # ('vgg',VGG), 
        # ('res',ResNet)
    ]
    RESULT = {}
    for mode in ['aff', 'col']:
        print('=======================')
        print('=========%s========='%mode)
        print('=======================')
        for name, model_func in model_frames:
            batch_size = 128
            dataloaders = make_dataloaders('usps', 'mnistm', batch_size)

            print(name)
            
            model = model_func().cuda()
            optimizer = make_optimizers(model)
            result, result2 = train_test_attack(model, trainloader, testloader, optimizer, criterion)
            print(result)
            print(result2)
            
            torch.save(model.state_dict(), 'ckpt/'+mode+'/'+name+'.pth')
            del model, optimizer
            torch.cuda.empty_cache()

            ## robust        
            robust_model = model_func().cuda()
            optimizer_r = make_optimizers(robust_model)
            result_c, result_c2 = train_test_attack(robust_model, trainloader, testloader, optimizer_r, criterion, cert=True, robust_loss=robust_loss)
            print(result_c)
            print(result_c2)

            torch.save(robust_model.state_dict(), 'ckpt/'+mode+'/'+name+'_cert.pth')
            del robust_model, optimizer_r
            torch.cuda.empty_cache()

            RESULT[name] = {'base_fgsm': result, 'base_pgd':result2, 'cert_fgsm':result_c, 'cert_pgd':result_c2}

        with open('ckpt/'+mode+'/'+'result.json','w') as f:
            json.dump(RESULT, f, sort_keys=True, indent=4)

    #plot

if __name__ == '__main__':
    main()

'''
model.load_state_dict(torch.load('ckpt/'+name+'.pth'))
result_c = train_test_attack(robust_model, trainloader, testloader, optimizer_r, criterion, cert=True, robust_loss=robust_loss_parallel)

'''