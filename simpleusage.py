import dataloader
from models.simpleconv import SimConv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import sys
import pandas as pd
from PIL import Image
import os.path
import glob
import numpy

def restorenet(img):
    transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    img = transform(img)
    img = (img*2)-1
    img = img.unsqueeze(0).to(torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'))
    print(img)
    net = torch.load('multiresult.pkl')
    result = net(img)
    print(result)
    return result.to('cpu').detach().numpy()

# def convertimages(picpath):
#     for pngfile in glob.glob(picpath):
#         img = Image.open(pngfile)
#         try:
#             out = restorenet(img)
#             print(out)
#         except Exception as e:
#             print(e) 

# convertimages(sys.argv[0])
print(restorenet(Image.open("reslut001.png").convert('RGB')))
