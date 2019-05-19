import torch
import torch.nn as nn
from torch.autograd import Function

class DaNN(nn.Module):
    def __init__(self):
        super(DaNN, self).__init__()
        self.main_features = nn.Sequential(
                nn.Conv2d(3, 64, 6, stride=2),
                # nn.BatchNorm2d(64), 
                # nn.MaxPool2d(2),
                nn.ReLU(True),
                nn.Conv2d(64, 50, 6, stride=2),
                # nn.BatchNorm2d(50),
                # nn.Dropout2d(),
                # nn.MaxPool2d(2),
                nn.ReLU(True),
            )
        self.classifier = nn.Sequential(
                nn.Linear(50*4*4, 100),
                # nn.BatchNorm1d(100),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(100,100),
                # nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100, 10),
                # nn.LogSoftmax(dim=1),
            )
        self.domain = nn.Sequential(
                nn.Linear(50*4*4, 100), 
                nn.BatchNorm1d(100),
                nn.ReLU(True),
                nn.Linear(100,2),
                # nn.LogSoftmax(dim=1)
            )
        self.reverse = ReverseLayerF
        self.alpha = 0

    def forward(self, images, mode='class'):
        # images = images.expand(images.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        features = self.main_features(images)
        features = features.view(-1, 50*4*4)

        reverse_feature = self.reverse.apply(features, self.alpha)
        if mode == 'class':
            output = self.classifier(features)
        else:
            output = self.domain(reverse_feature)
        
        return output

    def _set_alpha(self, alpha):
        self.alpha = alpha

    def one_seq_chunk(self):
        return nn.Sequential(*(list(self.main_features.children()) + [Flatten()] + list(self.classifier.children())))
    
    def update_chunk(self, chunk):
        self.main_features.load_state_dict(chunk[:3].state_dict())
        chunk_classifier = torch.nn.Sequential(* list(chunk[5:].children()))
        self.classifier.load_state_dict(chunk_classifier.state_dict())

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

if __name__ == '__main__':
    import torch
    model = DaNN()
    print(model)
    sample = torch.randn(128, 3, 28, 28)
    print('sample shape', sample.shape)
    a, b = model(sample, 0)
    print(a.shape)
    print(b.shape)
