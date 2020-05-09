import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.md3 import*

class ThreeVGCAudio(Dataset):

    def __init__(self, csv_path):
        csv_data = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csv_data)):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz)
        # Original sound size ([2, 221184])
        sound, sr = torchaudio.load(path, out=None, normalization=True)
        # take the average of the 2 stereo channels
        mean = torch.mean(sound, dim=0)
        # obtain 48 MFCC features, calculated with mel not db
        mfcc = torchaudio.transforms.MFCC(sr, log_mels=True, n_mfcc=48)(mean)
        #mel = torchaudio.transforms.MelSpectrogram(sr, n_mels=48)
        #keep only the first 1024
        sound_padded = mfcc[:, :1024].unsqueeze(0)
        return sound_padded, self.labels[index]

    def __len__(self):
        return len(self.file_names)



def train(model, epoch,device):
    model.train()
    train_losses = list()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(trainloader, 0):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        data = data.requires_grad_(True)
        output = model(data)
        loss = F.cross_entropy(output, target)  
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader),
                       100. * batch_idx / len(trainloader), loss))

    return sum(train_losses) / len(train_losses)


def test(model, epoch,device):
    model.eval()
    correct = 0
    total = 0
    test_accuracies = list()
    with torch.no_grad():

        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()

    test_accuracies = 100. * correct / total
    print('Test Accuracy on the {} Epoch: {} , [{}/{}]'.format(epoch, test_accuracies, correct, total))

    return test_accuracies


if __name__ == "__main__":

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')
    model = md3()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.88, nesterov=True)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular', cycle_momentum=True, base_lr=0.0001, max_lr=0.01)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2, verbose=True)

    
    train_csv_path = '~/3VGC/train_file_paths.csv'
    test_csv_path = '~/3VGC/test_file_paths.csv'
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
        lr_ = scheduler.get_lr()
        with open('md3d_sgd_big.csv', 'a') as result:
            print(f'{epoch},{str(acc)},{str(loss)}, {str(lr_[0])}', file = result)
    torch.save(model.state_dict(), 'md3_big.pth')