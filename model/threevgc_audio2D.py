import torch.nn as nn
import torch.nn.functional as F


class ThreeVGCAudio2D(nn.Module):
    def __init__(self):
        super(ThreeVGCAudio2D, self).__init__()
        # 1 input audio channel, 24 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, 96, 3, padding=1)
        self.fc1 = nn.Linear(2016, 128)
        self.fc2 = nn.Linear(128, 84)
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(96)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        # Max pooling over a (4, 4) window
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 4)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 4)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 3)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def threevgc2d():
    model = ThreeVGCAudio2D()
    return model
