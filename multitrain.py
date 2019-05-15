import dataloader
from models.multiconv import MutiConv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import numpy as np

def train(epochs = 30,batch = 8,device=torch.device('cpu')):
    train_data = dataloader.PairedDataset(mode="train",transform=transforms.ToTensor())
    val_data = dataloader.PairedDataset(mode="val",transform=transforms.ToTensor())
    train_loader = DataLoader(dataset= train_data,batch_size=batch,shuffle=True)
    val_loader = DataLoader(dataset= val_data,batch_size=batch)

    model = MutiConv().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.SmoothL1Loss()
    size = 1
    for epoch in range(epochs):
        print('epoch {}'.format(epoch+1))
        # start training
        train_loss = 0.
        train_acc = 0.
        for batch_x,batch_y in train_loader:
            batch_x,batch_y_n = Variable(batch_x.to(device)),Variable(torch.from_numpy(np.array([pointsize(batch_y,size)])).to(device))
            out = model(batch_x)
            #out=out.reshape(60)
            #print(out)
            #print(batch_y.qw)
            loss = loss_func(out.double(),batch_y_n)
            train_loss += loss.data
            pred = torch.max(out,1)[1].double()
            train_correct = (pred == Variable(torch.from_numpy(np.array([batch_y])).to(device))).sum()
            train_acc += train_correct.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))


        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x.to(device), volatile=True),Variable(torch.from_numpy(np.array([pointsize(batch_y,size)])).to(device), volatile=True)
            out = model(batch_x)
            loss = loss_func(out.double(), batch_y)
            eval_loss += loss.data
            pred = torch.max(out, 1)[1].double()
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_data)), eval_acc / (len(val_data))))
        #if epoch%5 == 0:
            #torch.save(model,'multiepoch'+epoch+'.pkl')
    torch.save(model,'multiresult.pkl')


def pointsize(x,size):
    result = []
    for k in x:
        a = k*(10**size)
        result.append(int(a)/(10**size))
    return result

device_2 = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
train(30,1,device=device_2)
