import dataloader
from models.simpleconv import SimConv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import sys
import cv2
import pandas as pd
from PIL import Image
import os.path
import glob
import numpy
import csv
from PIL import Image

source_video = "test2.mp4"
#load the video
cap = cv2.VideoCapture(source_video)
c = 0
rval = cap.isOpened()

def restorenet(img):

    transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    img = transform(img)
    #img = (img*2)-1
    img = img.unsqueeze(0).to(torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'))
    net = torch.load('multiresult.pkl')
    result = net(img)
    return result

while rval:
    c = c+1
    rval,frame = cap.read()
    if rval:
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        img = img.resize((256,256),Image.ANTIALIAS)
        img = img.resize((600,360),Image.ANTIALIAS)
        box = (150,0,450,360)
        img = img.crop(box)
        img = img.resize((256,256),Image.ANTIALIAS)
        result = restorenet(img)
        result_csv = result.to('cpu').detach().numpy().reshape(1,60)
        joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                   'Neck_x', 'Neck_y', 'Neck_z', 
                   'Head_x', 'Head_y', 'Head_z', 
                   'Nose_x', 'Nose_y', 'Nose_z', 
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z',
                   'hip.Center_x','hip.Center_y','hip.Center_z']
        joints_export = pd.DataFrame(result_csv, columns=joints_names)
        joints_export.index.name = 'frame'
            
        joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
        joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1
        joints_export.to_csv("output/test"+str(c)+".csv")
        #this part we have got the wanted result,now let combine the results into bvh
        path = 'output/'                   
        all_files = glob.glob(os.path.join(path, "*.csv"))
        df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
        concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
        concatenated_df['frame'] = concatenated_df.index+1
        concatenated_df.to_csv("result.csv", index=False)
    else:
        break
#here the csvs are combined,now comes the bvh convert



