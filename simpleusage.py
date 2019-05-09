import dataloader
from models.simpleconv import SimConv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable

def restorenet(img):
    net = torch.load('simpleresult.pkl')
    result = net(img)
    print(result)