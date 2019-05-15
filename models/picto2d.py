import torch
import torch.nn as nn

Norm = nn.BatchNorm2d

class Picto2d(torch.nn.Module):
    def __init__(self):
        super(Picto2d,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            Norm(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64, 3, 1, 1),
            Norm(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            Norm(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.vgg_layer = nn.Sequential(
            VGG(128)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 40)
        )
    def forward(self,x):
        #img preprocess
        result = self.conv(x)#32*32*64
        #vgg part
        result = self.vgg_layer(result)#4*4*128
        #turn into [20,2]
        res = result.view(result.size(0), -1)
        result = self.dense(res)
        return result

class VGG(nn.Module):#input(x.w*x.h*128)output(x.w/8*128)
    def __init__(self,in_channels = 128):
        super(VGG,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,64,3,1,1)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.norm1 = Norm(64)
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.norm2 = Norm(128)
        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.norm3 = Norm(256)
        self.conv4 = nn.Conv2d(256,512,3,1,1)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.norm4 = Norm(512)
        self.conv5 = nn.Conv2d(512,256,3,1,1)
        self.norm5 = Norm(256)
        self.conv6 = nn.Conv2d(256,128,3,1,1)
        self.norm6 = Norm(128)
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        #block 1(x.w*x.h*128)
        feature = torch.relu(x)
        feature = self.conv1(feature)
        feature = self.norm1(feature)
        feature = torch.relu(feature)
        feature = self.conv1_2(feature)
        feature = self.norm1(feature)
        feature = torch.relu(feature)
        feature = self.pool(feature)
        
        #block2(x.w/2*64)
        feature = self.conv2(feature)
        feature = self.norm2(feature)
        feature = torch.relu(feature)
        feature = self.conv2_2(feature)
        feature = self.norm2(feature)
        feature = torch.relu(feature)
        feature = self.pool(feature)

        #block3(x.w/4*128)
        feature = self.conv3(feature)
        feature = self.norm3(feature)
        feature = torch.relu(feature)
        feature = self.conv3_2(feature)
        feature = self.norm3(feature)
        feature = torch.relu(feature)
        feature = self.conv3_2(feature)
        feature = self.norm3(feature)
        feature = torch.relu(feature)
        feature = self.conv3_2(feature)
        feature = self.norm3(feature)
        feature = torch.relu(feature)
        feature = self.pool(feature)
        #block4(x.w/8*256)
        feature = self.conv4(feature)
        feature = self.norm4(feature)
        feature = torch.relu(feature)
        feature = self.conv4_2(feature)
        feature = self.norm4(feature)
        feature = torch.relu(feature)
        #additional none vgg layer(x.w/8*512)
        feature = self.conv5(feature)
        feature = self.norm5(feature)
        feature = torch.relu(feature)
        feature = self.conv6(feature)
        feature = self.norm6(feature)
        feature = torch.relu(feature)

        #(x.w/8*128)
        return feature
