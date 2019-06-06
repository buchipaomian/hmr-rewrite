import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.cuda
import torchvision.transforms as transforms
import dataloader2

def make_model():
    #model = models.vgg16(pretrained = True)
    model = models.resnet50(pretrained = True)
    #model.classifier[6] = nn.Linear(4096,40)
    model = nn.Sequential(
        model,
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 60))
    model = model.eval()
    return model
transformation = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
def train(epochs = 30,batch = 8,device=torch.device('cpu')):
    train_data = dataloader2.PictoTwoDataset(mode="train",transform=transformation)
    val_data = dataloader2.PictoTwoDataset(mode="val",transform=transforms.ToTensor())
    train_loader = DataLoader(dataset= train_data,batch_size=batch)
    val_loader = DataLoader(dataset= val_data,batch_size=batch)

    model = make_model().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    size = 3
    for epoch in range(epochs):
        #convl_weight = model.conv[0].weight
        #convlweight2 = model.dense1[0].weight
        #print(convlweight2)
        #print(convl_weight)
        print('epoch {}'.format(epoch+1))
        # start training
        train_loss = 0.
        train_acc = 0.
        for count,(batch_x,batch_y,size_z,file_id) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x,batch_y = batch_x.to(device),torch.from_numpy(np.array([pointsize(batch_y,size)])).to(device)
            #print("now processing")
            #print(file_id)
            #print("this is x")
            #print(batch_x)
            out = model(batch_x)
            #print("this is z,which is normed")
            #print(z)
            #print("this is z1,which is normal conved")
            #print(z1)
            #print("this is k,which is vgg output")
            #print(k)
            #out=out.reshape(60)
            #print(out)
            #print(batch_y.qw)
            loss = loss_func(out,batch_y.float())
            train_loss += loss.data
            #pred = torch.max(out,1)[1].double()
            #train_correct = (pred == batch_y).sum()
            #train_acc += train_correct.data
            loss.backward()
            optimizer.step()
            if count%400 == 0:
                print(out)


        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y,size_z,file_id in val_loader:
            batch_x, batch_y = batch_x.to(device),torch.from_numpy(np.array([pointsize(batch_y,size)])).to(device)
            with torch.no_grad():
                out = model(batch_x)
            #print(out)
            loss = loss_func(out, batch_y.float())
            eval_loss += loss.data
            #pred = torch.max(out, 1)[1].double()
            #num_correct = (pred == batch_y).sum()
            #eval_acc += num_correct.data
        print('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
        print('Test Loss: {:.6f}'.format(eval_loss / (len(val_data))))
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
