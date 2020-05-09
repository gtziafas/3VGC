import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.md1 import *
from model.md2 import *


def train(model, epoch, device):
    model.train()
    train_losses = list()
    for batch_idx, (data, target) in enumerate(trainloader, 0):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        data = data.requires_grad_(True)  # set requires_grad to True for training
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader),
                       100. * batch_idx / len(trainloader), loss))

    return sum(train_losses) / len(train_losses)

def test(model, epoch, device):
    model.eval()
    correct = 0
    total = 0
    test_accuracies = list()
    with torch.no_grad():

        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            print(target.shape)
            output = model(data)
            print(output.shape)
            _, pred = torch.max(output, 1) 
            print(pred.shape, pred)
            total += target.size(0)
            correct += (pred == target).sum().item()

    test_accuracies = 100. * correct / total
    print('Test Accuracy on Epoch: {} is {}/{} {} '.format(epoch,correct,total,test_accuracies))

    return test_accuracies
   

if __name__ == "__main__":

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')
    model = md1()
    #Uncomment to train and test on MD2 
    #model = md2() 
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.88, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2, verbose=True)
    
    n_fold = 2
    train_ = ['train'+str(i)+'.csv' for i in range(n_fold)]
    test_ = ['test'+str(i)+'.csv' for i in range(n_fold)]
    
    
    for i in range(n_fold):
        train_csv_path = train_[i]
        test_csv_path = test_[i]
        train_set = ThreeVGCAudio(csv_path=train_csv_path)
        test_set = ThreeVGCAudio(csv_path=test_csv_path)
        print("Train set size: " + str(len(train_set)))
        print("Test set size: " + str(len(test_set)))

        kwargs = {'num_workers': 10,
                  'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)
        loss, acc = [], []
        log_interval = 20

        for epoch in range(1, 41):
            loss = train(model, epoch, device)
            acc = test(model, epoch, device)
            scheduler.step()
            with open('md1_adam.csv', 'a') as result:
                print(f'{epoch},{str(acc)},{str(loss)}', file = result)
        torch.save(model.state_dict(), 'md1_adam.pth')

        del trainloader
        del testloader
        del train_set