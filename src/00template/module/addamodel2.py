import torch.nn.functional as F
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, x):
        x = x.expand(x.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        conv_out = self.encoder(x)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat

class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, x):
        x = x.expand(x.shape[0], 3, 28, 28)  ### for usps 1 --> 3
        conv_out = self.encoder(x)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 10)


    def forward(self, feat):
        out = F.dropout(F.relu(feat), training=self.training)
        out = F.dropout(F.relu(self.fc1(out)))
        out = F.softmax(self.fc2(out), dim=1)
        return out

class Domain_teller(nn.Module):

    def __init__(self):
        super(Domain_teller, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out

class ADDA(nn.Module):
    def __init__(self):
        super(ADDA, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.tencoder = Encoder2()
        self.teller = Domain_teller()

    def forward(self, x, mode='source'):
        if mode == 'source':
            x = self.encoder(x)
            x = self.classifier(x)
        elif mode =='domains':
            x = self.encoder(x)
            x = self.teller(x)
        elif mode =='domain':
            x = self.tencoder(x)
            x = self.teller(x)
        elif mode =='target':
            x = self.tencoder(x)
            x = self.classifier(x)
        return x

    def target_load_source(self):
        self.tencoder.load_state_dict(self.encoder.state_dict())

    def save(self, filepath):
        state = {
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'tencoder': self.tencoder.state_dict(),
            'teller': self.teller.state_dict()
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.encoder.load_state_dict(state['encoder'])
        self.classifier.load_state_dict(state['classifier'])
        self.tencoder.load_state_dict(state['tencoder'])
        self.teller.load_state_dict(state['teller'])

    def load_pretrain(self, filepath):
        state = torch.load(filepath)
        self.encoder.load_state_dict(state['encoder'])
        self.classifier.load_state_dict(state['classifier'])
        


if __name__ == '__main__':
    import torch
    adda = ADDA()
    print(adda)
    sample = torch.randn(128, 3, 28, 28)
    print('sample shape', sample.shape)
    # feature = adda.encoder(sample)
    # output = adda.classifier(feature)
    # dom = adda.teller(feature)
    # print(feature.shape)
    # print(output.shape)
    # print(dom.shape)