import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.cuda
import torchvision.transforms as transforms
import dataloader
import model.Unet as unet
import utils.losses as Losses

#def the transform of image(here convert into 224*224)
transformation = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])

#this is the train method,net is Unet,loss is unetloss
def train(epochs = 30,batch = 8,device=torch.device('cpu')):
    train_data = dataloader.PairedDataset(mode="train",transform=transformation)
    val_data = dataloader.PairedDataset(mode="val",transform=transforms.ToTensor())
    train_loader = DataLoader(dataset= train_data,batch_size=batch)
    val_loader = DataLoader(dataset= val_data,batch_size=batch)

    model = unet.Unet().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = Losses.UnetLoss(device=device)
    size = 3
    for epoch in range(epochs):
        model.train()    
    #convl_weight = model.conv[0].weight
        #convlweight2 = model.dense1[0].weight
        #print(convlweight2)
        #print(convl_weight)
        print('epoch {}'.format(epoch+1))
        # start training
        train_loss = 0.
        # train_acc = 0.
        for count,(batch_x,batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            if batch_y == -1:
                continue
            batch_x= batch_x.to(device)
            out = model(batch_x)
            loss = loss_func(out,batch_y)
            train_loss += loss.data
            loss.backward()
            optimizer.step()
            if count%4000 == 0:
                print(out)
                print(batch_y)


        # model.eval()
        # eval_loss = 0.
        # eval_acc = 0.
        # for batch_x, batch_y,size_z,file_id in val_loader:
        #     batch_x, batch_y = batch_x.to(device),torch.from_numpy(np.array([pointsize(batch_y,size)])).to(device)
        #     with torch.no_grad():
        #         out = model(batch_x)
        #     print(out)
        #     loss = loss_func(out, batch_y.float())
        #     eval_loss += loss.data
            #pred = torch.max(out, 1)[1].double()
            #num_correct = (pred == batch_y).sum()
            #eval_acc += num_correct.data
        print('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
        #print('Test Loss: {:.6f}'.format(eval_loss / (len(val_data))))
        # if train_loss/eval_loss > 10:
        #     size+=1
        if epoch%3 == 0:
            torch.save(model,'multiepoch'+str(epoch)+'.pkl')
    torch.save(model,'multiresult.pkl')


def pointsize(x,size):
    result = []
    for k in x:
        a = k*(10**size)
        result.append(int(a)/(10**size))
    return x

device_2 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
train(30,1,device=device_2)
