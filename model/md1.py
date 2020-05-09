import torch.nn as nn
import torch.nn.functional as F

#Implementation from the paper: https://arxiv.org/pdf/1610.00087.pdf
#after M5 architecture 

class MD1(nn.Module):
    def __init__(self):
        super(MD1, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 6)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim = 1)
        return x

def md1():
    model = MD1()
    return model

