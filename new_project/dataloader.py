import torch
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data.dataloader as DataLoader
import tools
from torchvision import transforms
import os
import glob
from PIL import Image
import pandas as pd
import json

class PairedDataset(Dataset):
    """
    Composed of
    (origanal image,output json)
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
            img_root = os.path.join(root,"train/image")
            json_root = os.path.join(root,"train/json_body")
        elif mode == "val":
            img_root = os.path.join(root,"val/image")
            json_root = os.path.join(root,"val/json_body")
        else:
            raise ValueError('Invalid Dataset. Pick among (train, val)')
        self.is_train = (mode == 'train')
        self.transform = transform
        self.image_files = glob.glob(os.path.join(img_root, '*.jpg'))
        self.json_files = glob.glob(os.path.join(json_root, '*.json'))
        self.color_cache = {}

        if len(self.image_files) == 0:
            # no png file, use jpg
            self.image_files = glob.glob(os.path.join(img_root, '*.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Niko Dataset Get Item
        @param index: index
        Returns:
            image
            paired json
        """
        img_filename = self.image_files[index]
        json_filename = self.json_files[index]
        # file_id = img_filename.split('/')[-1][:-4]


        image = Image.open(img_filename)
        jsonfile = pd.read_json(json_filename)
        k = jsonfile.to_dict()
        try:
            joint = k["bodies"][0]["joints19"]
        except:
            return self.transform(image),-1
        joint_result = []
        for i in range(len(joint)):
            if i % 4 != 3:
                joint_result.append(joint[i])
        if self.transform is not None:
            image = self.transform(image)
        return image,joint_result