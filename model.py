import torch
import torch.nn as nn

# Model design
class ASVCNN(nn.Module):
    def __init__(self, input_size=1, num_classes = 1):
        super(ASVCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # might have to change the pooling kernel stride to an overlapping one after the training
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layer2 = nn.Sequential(
            #kernel size remains the same, hence the padding and striding as well
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #changing the pooling kernel size so that the layer 1 feautremap are made in a higher resolution feature map.
            #no pooling around mfcc axis, only pooling along time frame axis
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        #classification layers
        self.linear1 = nn.Sequential(
            nn.Linear(9600,128),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        # print('layer1: ',x.shape)
        x = self.layer2(x)
        # print('layer 2',x.shape)
        x = self.layer3(x)
        # print('layer 3',x.shape)

        #flattening x
        x = torch.flatten(x, start_dim=1)
        # print('after flattening',x.shape)

        x = self.linear1(x)
        # print('linear 1:', x.shape)
        x = self.linear2(x)
        # print('linear 2:',x.shape)
        return x