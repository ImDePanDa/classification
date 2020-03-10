import os
import sys
import torch
import torch.nn as nn 
import torch.optim as optim

from torch.utils.data import DataLoader
# from nets import foodnet
from resnet import ResNet18
from datasets import csvdataset
from sklearn.metrics import confusion_matrix
from utils import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

# net = foodnet().cuda() if torch.cuda.is_available() else foodnet()
net = ResNet18().cuda() if torch.cuda.is_available() else ResNet18()
optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)
loss_fc = nn.CrossEntropyLoss()

trainCsv = ["train0", "train1", "train2", "train3", "train4", "train5", "train6", "train7"]
# trainCsv = ["train0"]
train_dataset = csvdataset(root=r"./FoodChallenge", csvnames=trainCsv)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=1, shuffle=True)

valCsv = ["train8", "train9"]
val_dataset = csvdataset(root=r"./FoodChallenge", csvnames=valCsv)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=1, shuffle=True)

def train(epoch):
    total_step, total_loss, total_acc, total_num = len(train_dataloader), 0, 0, 0
    for i, (img, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        img = img.cuda() if torch.cuda.is_available() else img
        label = label.cuda() if torch.cuda.is_available() else label
        output = net(img)
        loss = loss_fc(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        val, index = torch.max(output, 1)
        total_acc += label.eq(index.data).sum()
        total_num += len(label)
        print('Train - Epoch: {} | Step: {}/{} | Loss: {:05f} | Acc: {:05f}'.format(epoch, i+1, total_step, total_loss/total_num, int(total_acc)/total_num))

def val(epoch):
    total_num, total_loss, total_acc = 0, 0, 0
    for i, (img, label) in enumerate(val_dataloader):
        img = img.cuda() if torch.cuda.is_available() else img
        label = label.cuda() if torch.cuda.is_available() else label
        pred = net(img)
        total_loss += loss_fc(pred, label).data
        total_num += len(label)
        val, index = torch.max(pred, 1)
        index, label = index.detach().cpu(), label.detach().cpu()
        total_index = index if i==0 else torch.cat([total_index, index])
        total_label = label if i==0 else torch.cat([total_label, label])
        ConfsMat = confusion_matrix(total_label, total_index)
        total_acc += label.eq(index.data).sum()
    print('Val - Epoch: {} | Acc: {:05f} | Loss: {:05f}'.format(epoch, int(total_acc)/total_num, total_loss/total_num))
    print("ConfusionMat is:\n {}".format(ConfsMat))
    return int(total_acc)/total_num, total_loss/total_num

if __name__ == '__main__':
    sys.stdout = Logger()
    acc, loss = 0, 100
    for epoch in range(0, 200):            
        train(epoch)
        acc_, loss_ = val(epoch)
        if acc_>acc and loss_<loss:
            torch.save(net.state_dict(), './models/epoch_{}_acc_{:.3f}.pth'.format(epoch, acc_))
            acc = acc_
            loss = loss_