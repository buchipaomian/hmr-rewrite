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

class PairedDataset(Dataset):
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

        # if self.color_histogram:
        #     # build colorgram tensor
        #     color_info = self.color_cache.get(file_id, None)
        #     if color_info is None:
        #         with open(
        #                 os.path.join('./data/colorgram', '%s.json' % file_id).replace('\\', '/'),
        #                 'r') as json_file:
        #             # load color info dictionary from json file
        #             color_info = json.loads(json_file.read())
        #             self.color_cache[file_id] = color_info
        #     colors = make_colorgram_tensor(color_info)

        image = Image.open(img_filename)
        csvfile = pd.read_csv(csv_filename)
        k = csvfile.to_dict()
        del k['frame']
        joint = []
        for item in k:
            joint.append(k[item][0])
        return image,joint,file_id
        # # default transforms, pad if needed and center crop 512
        # width_pad = self.size - image_width // 2
        # if width_pad < 0:
        #     # do not pad
        #     width_pad = 0

        # height_pad = self.size - image_height
        # if height_pad < 0:
        #     height_pad = 0

        # # padding as white
        # padding = transforms.Pad((width_pad // 2, height_pad // 2 + 1,
        #                           width_pad // 2 + 1, height_pad // 2),
        #                          (255, 255, 255))

        # # use center crop
        # crop = transforms.CenterCrop(self.size)

        # imageA = padding(imageA)
        # imageA = crop(imageA)

        # imageB = padding(imageB)
        # imageB = crop(imageB)

        # if self.transform is not None:
        #     imageA = self.transform(imageA)
        #     imageB = self.transform(imageB)

        # # scale image into range [-1, 1]
        # imageA = scale(imageA)
        # imageB = scale(imageB)
        # if not self.color_histogram:
        #     return imageA, imageB
        # else:
        #     return imageA, imageB, colors
