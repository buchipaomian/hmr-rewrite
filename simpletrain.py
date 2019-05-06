import dataloader
from models.simpleconv import SimConv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable

def train(epochs = 30,batch = 8,device=torch.device('cpu')):
    train_data = dataloader.PairedDataset(mode="train",transform=transforms.ToTensor())
    val_data = dataloader.PairedDataset(mode="val",transform=transforms.ToTensor())
    train_loader = DataLoader(dataset= train_data,batch_size=batch,shuffle=True)
    val_loader = DataLoader(dataset= val_data,batch_size=batch)

    model = SimConv().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in epochs:
        print('epoch {}'.format(epoch+1))
        # start training
        train_loss = 0.
        train_acc = 0.
        for batch_x,batch_y in train_loader:
            batch_x,batch_y = Variable(batch_x.to(device)),Variable(batch_y.to(device))
            out = model(batch_x)
            loss = loss_func(out,batch_y)
            train_loss += loss.data[0]
            pred = torch.max(out,1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))


        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x.to(device), volatile=True), Variable(batch_y.to(device), volatile=True)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data[0]
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_data)), eval_acc / (len(val_data))))
    torch.save(model,'simpleresult.pkl')
device_2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
train(30,8,device=device_2)