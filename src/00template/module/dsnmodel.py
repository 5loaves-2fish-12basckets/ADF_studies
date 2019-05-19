import torch
from torch.autograd import Function

class private_encoder2(torch.nn.Module):

    def __init__(self, arg):
        super(private_encoder2, self).__init__()
        self.main = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2,stride=2),
                torch.nn.Conv2d(32,64, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2, stride=2),
            )
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(7*7*64, arg.z_dim),
                torch.nn.ReLU(True)
            )
    def forward(self, x):
        x = x.expand(x.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        x = self.main(x)
        x = x.view(x.shape[0], -1) # 7*7*64
        x = self.fc(x)
        return x

class private_encoder(torch.nn.Module):

    def __init__(self, arg):
        super(private_encoder, self).__init__()
        self.main = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2,stride=2),
                torch.nn.Conv2d(32,64, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2, stride=2),
            )
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(7*7*64, arg.z_dim),
                torch.nn.ReLU(True)
            )
    def forward(self, x):
        x = x.expand(x.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        x = self.main(x)
        x = x.view(x.shape[0], -1) # 7*7*64
        x = self.fc(x)
        return x

class shared_encoder(torch.nn.Module):

    def __init__(self, arg):
        super(shared_encoder, self).__init__()
        self.main = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2,stride=2),
                torch.nn.Conv2d(32,48, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(2, stride=2),
            )
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(7*7*48, arg.z_dim),
                torch.nn.ReLU(True)
            )
    def forward(self, x):
        x = x.expand(x.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        x = self.main(x)
        x = x.view(x.shape[0], -1) # 7*7*48
        x = self.fc(x)
        return x

class shared_decoder(torch.nn.Module):
    """docstring for shared_decoder"""
    def __init__(self, arg):
        super(shared_decoder, self).__init__()
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(arg.z_dim, 12*7*7),
                torch.nn.ReLU(True)
            )
        self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 5, padding=2),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(16, 16, 5, padding=2),
                torch.nn.ReLU(True),
                Upsample(),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(16, 3, 3, padding=1),

            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 3, 14, 14)
        x = self.conv(x)
        return x

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)


class DSN(torch.nn.Module):
    """docstring for DSN"""
    def __init__(self, arg):
        super(DSN, self).__init__()
        self.arg = arg
        self.priv_targ_encoder = private_encoder(arg)
        self.priv_sour_encoder = private_encoder2(arg)
        self.shared_encoder = shared_encoder(arg)
        self.shared_decoder = shared_decoder(arg)
        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(arg.z_dim, 100),
                torch.nn.ReLU(True),
                torch.nn.Linear(100, arg.n_classes)
            )
        self.domain = torch.nn.Sequential(
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(True),
                torch.nn.Linear(100, 2)
            )
        
    def forward(self, images, mode='source', scheme='all', alpha=0):

        if mode == 'source':
            private_code = self.priv_sour_encoder(images)
        elif mode == 'target':
            private_code = self.priv_targ_encoder(images)

        shared_code = self.shared_encoder(images)
        shared_code2 = shared_code.clone()
        shared_code3 = shared_code.clone()

        rev_shared_code = ReverseLayerF.apply(shared_code2, alpha)
        domain_label = self.domain(rev_shared_code)

        if mode == 'source':
            class_label = self.classifier(shared_code3)

        if scheme =='share':
            union_code = shared_code.clone()
        elif scheme == 'all':
            union_code = private_code + shared_code
        elif scheme == 'private':
            union_code = private_code.clone()

        recreated = self.shared_decoder(union_code)

        if mode == 'source':
            result = [private_code, shared_code, domain_label, class_label, recreated]
        elif mode == 'target':
            result = [private_code, shared_code, domain_label, recreated]
        
        return result 

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

##losses

class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, targ):
        diff = torch.add(pred, -targ)
        return torch.sum(diff.pow(2))/torch.numel(diff.data)

class scale_inv_MSE(torch.nn.Module):
    def __init__(self):
        super(scale_inv_MSE, self).__init__()

    def forward(self, pred, targ):
        diff = torch.add(pred, -targ)
        return torch.sum(diff).pow(2) / torch.numel(diff.data)**2

class DifferenceLoss(torch.nn.Module):
    def __init__(self):
        super(DifferenceLoss, self).__init__()

    def process(self, code):
        code = code.view(code.size(0), -1)
        code_norm = torch.norm(code, p=2, dim=1, keepdim=True).detach()
        code_l2 = code.div(code_norm.expand_as(code) + 1e-6)
        return code_l2

    def forward(self, a,b):
        a_l2 = self.process(a)
        b_l2 = self.process(b)

        return torch.mean((a_l2.t().mm(b_l2)).pow(2))


if __name__ == '__main__':
    class model_param(object):
        def __init__(self):
            self.z_dim = 100
            self.n_classes = 10

    args = model_param()
    model = DSN(args)
    # print(model)
