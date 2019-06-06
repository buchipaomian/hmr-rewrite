import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,input):
        return self.conv(input)
class Unet(torch.nn.Module):
    def __init__(self,in_ch = 3):
        """
        the input is 224*224*3
        with a u-net convert it into a model type which is 112*112*112
        """
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)        #224*224*3 to 112*112*64
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)        #112*112*64 to 56*56*128
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)        #56*56*128 to 28*28*256
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)        #28*28*256 to 14*14*512
        self.conv5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)        #14*14*512 to 7*7*1024
        self.conv6 = DoubleConv(1024, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)#7*7*1024 to 14*14*512
        self.conv7 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)#28*28*256
        self.conv8 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)#56*56*128
        self.conv9 = DoubleConv(256, 224)
        self.up9 = nn.ConvTranspose2d(224, 112, 2, stride=2)#112*112*112
        self.conv9 = DoubleConv(112, 112)
        # self.conv10 = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        c1 = self.conv1(x)#224*224*64
        p1 = self.pool1(c1)#112
        c2 = self.conv2(p1)#112*112*128
        p2 = self.pool2(c2)#56
        c3 = self.conv3(p2)#56*56*256
        p3 = self.pool3(c3)#28
        c4 = self.conv4(p3)#28*28*512
        p4 = self.pool4(c4)#14
        c5 = self.conv5(p4)#14*14*1024
        p5 = self.pool5(c5)#here turn to 7*7*1024
        c6 = self.conv6(p5)#7*7*1024
        up_6 = self.up6(c6)#14*14*512
        merge6 = torch.cat([up_6, p4], dim=1)#14*14*1024
        c7 = self.conv7(merge6)#14*14*512
        up_7 = self.up7(c7)#28*28*256
        merge7 = torch.cat([up_7, p3], dim=1)#28*28*512
        c8 = self.conv8(merge7)#28*28*256
        up_8 = self.up8(c8)#56*56*128
        merge8 = torch.cat([up_8, p2], dim=1)#56*56*256
        #here about to change something,since i want the output to be 112*112*112
        c8 = self.conv8(merge8)#56*56*224
        up_9 = self.up9(c8)#112*112*112
        # merge9 = torch.cat([up_9, c1], dim=1)
        # c9 = self.conv9(merge9)
        # c10 = self.conv10(c9)
        c9 = self.conv9(up_9)#112*112*112
        out = nn.ReLU()(c9)
        return out