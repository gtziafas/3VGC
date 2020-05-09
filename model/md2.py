import torch.nn as nn
import torch.nn.functional as F

#Implementation from the paper: https://arxiv.org/abs/1904.08990

class MD2(nn.Module):
    def __init__(self):
        super(MD2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=8, stride= 2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(3) #input should be 128x3 so this outputs a 128x1
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 6)
        
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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x

def md2():
    model = MD2()
    return model
