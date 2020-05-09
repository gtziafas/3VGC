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
        self.file_names = []
        self.labels = []
        for i in range(0, len(csv_data)):
            self.file_names.append(csv_data.iloc[i, 0])
            self.labels.append(csv_data.iloc[i, 1])

    def __getitem__(self, index):
        path = self.file_names[index]
        sound, sr = torchaudio.load(path, out=None, normalization=True)
        mean = torch.mean(sound, dim=0)
        mfcc = torchaudio.transforms.MFCC(sr, log_mels=True, n_mfcc=48)(mean)
        sound_padded = mfcc[:, :1024].unsqueeze(0)

        return sound_padded, self.labels[index]

    def __len__(self):
        return len(self.file_names)


def test(model, device):
    model.eval()
    correct = 0
    total = 0
    test_accuracies = list()
    classes = ('music', 'drama', 'sports', 'tv',
           'vlog', 'doc')
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    with torch.no_grad():

        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            c = (pred == target).squeeze()
            total += target.size(0)
            correct += (pred == target).sum().item()
            for i in range(6):
                targets = target[i]
                class_correct[targets] += c[i].item()
                class_total[targets] += 1

    test_accuracies = 100. * correct / total
    print('Test Accuracy: {} , [{}/{}]'.format(test_accuracies, correct, total))
    for i in range(6):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return test_accuracies


if __name__ == "__main__":

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')
    model = md3()
    #can be used also for 1D case, loading 1D models and using 1D Audioloader
    tensor_dict = model.load_state_dict(torch.load('model/mod2d_SGD_big_ds_88.pth', map_location=device))
    model.to(device)
    test_csv_path = 'test_file_paths.csv'
    test_set = ThreeVGCAudio(csv_path=test_csv_path)
    print("Test set size: " + str(len(test_set)))

    kwargs = {'num_workers': 10,
              'pin_memory': True} if device == 'cuda' else {}

    testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, **kwargs)
    epoch=1
    acc = test(model, device)