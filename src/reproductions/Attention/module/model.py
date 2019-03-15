import torch
import torch.nn as nn

## give up cifar10 just use vgg or something else

class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(args.img_channel_num,6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(9720,84),
            # nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
            nn.LogSoftmax(dim=-1)
            )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def save(self, filepath):
        state = {
            'conv': self.conv.state_dict(),
            'fc': self.fc.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.conv.load_state_dict(state['conv'])
        self.fc.load_state_dict(state['fc'])

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

class basicConvNet(nn.Module):
    def __init__(self):
        super(basicConvNet, self).__init__()

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()