import torch
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data.dataloader as DataLoader
import tools
from torchvision import transforms
import os
import glob
import csv
from PIL import Image
import pandas as pd

# class test_dataset(Dataset.Dataset):
#     # init
#     def __init__(self, Data, Label):
#         self.Data = Data
#         self.Label = Label

#     # return the size of dataset
#     def __len__(self):
#         return len(self.Data)

#     # get the content and label
#     def __getitem__(self, index):
#         data = torch.Tensor(self.Data[index])
#         label = torch.Tensor(self.Label[index])
#         if torch.cuda.is_available():
#             data = data.cuda()
#             label = label.cuda()
#         return data, label

class PictoTwoDataset(Dataset):
    """
    Composed of
    (origanal image,output csv)
    """

    def __init__(self,
                 root='./data/',
                 mode='train',
                 transform=None,
                 ):
        """
        @param root: data root
        @param mode: set mode (train, test, val)
        @param transform: Image Processing
        """

        if mode == 'train':
            img_root = os.path.join(root,"train_image")
            csv_root = os.path.join(root,"train_csv")
        elif mode == "val":
            img_root = os.path.join(root,"val_image")
            csv_root = os.path.join(root,"val_csv")
        else:
            raise ValueError('Invalid Dataset. Pick among (train, val)')
        self.is_train = (mode == 'train')
        self.transform = transform
        self.image_files = glob.glob(os.path.join(img_root, '*.png'))
        self.csv_files = glob.glob(os.path.join(csv_root, '*.csv'))
        self.color_cache = {}

        if len(self.image_files) == 0:
            # no png file, use jpg
            self.image_files = glob.glob(os.path.join(img_root, '*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Niko Dataset Get Item
        @param index: index
        Returns:
            image
            paired csv
            img name
        """
        img_filename = self.image_files[index]
        csv_filename = self.csv_files[index]
        file_id = img_filename.split('/')[-1][:-4]

        image = Image.open(img_filename)
        image = image.resize((224,224))
        csvfile = pd.read_csv(csv_filename)
        k = csvfile.to_dict()
        del k['frame']
        jointA = []
        jointB = []
        count = 1
        for item in k:
            if count%3 == 0:
                jointB.append(-k[item][0])
            elif count%3 == 1:
                jointA.append(k[item][0])
            else:
                jointA.append(-k[item][0])
            count += 1
        if self.transform is not None:
            image = self.transform(image)
        return image,jointA,jointB,file_id
