import torch
import torch.nn as nn

class MutiConv(torch.nn.Module):
    def __init__(self):
        super(MutiConv,self).__init__()
        #256 to 128
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        #128to64
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        #64to32
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        # 32to16
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        #16to8
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        #8to4
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        # 4to2
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        #2to1
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1024 * 1 * 1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 60)
        )