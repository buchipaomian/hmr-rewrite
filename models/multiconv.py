import torch
import torch.nn as nn

Norm = nn.BatchNorm2d

class MulticonvGenerator(nn.Module):
    """
    this is the generator of basic model
    """
    def __init__(self):
        super(MulticonvGenerator, self).__init__()

        self.bias = True
        self.dim = 3
    
    def _down_sample(self):
        layers = nn.ModuleList()
        #224
        layers.append(McNetDownSample(self.dim, self.dim * 2, self.bias))
        #112
        layers.append(McNetDownSample(self.dim*2, self.dim * 4, self.bias))
        #56
        layers.append(McNetDownSample(self.dim*4, self.dim * 8, self.bias))
        #28
        layers.append(McNetDownSample(self.dim*8, self.dim * 16, self.bias))
        #14
        layers.append(McNetDownSample(self.dim*16, self.dim * 32, self.bias))
        #7*7*192
        layers.append(McNetDownSample(self.dim*32, self.dim * 64, self.bias))
        
        return layers


class McNetDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(McNetDownSample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = Norm(out_channels)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        if in_channels == out_channels:
            self.channel_map = nn.Sequential()
        else:
            self.channel_map = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        feature = torch.relu(x)

        feature = self.conv1(feature)
        feature = self.norm1(feature)

        feature = torch.relu(feature)
        feature = self.conv2(feature)
        feature = self.norm2(feature)

        connection = feature + self.channel_map(x)
        feature, idx = self.pool(connection)
        return feature, connection, idx