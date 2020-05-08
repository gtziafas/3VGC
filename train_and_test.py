import csv
from datetime import datetime

import torch.optim as optim
from torch.utils.data import Dataset

from model.audioloader import *
from model.threevgc_audio1D import *
from model.threevgc_audio2D import *


def train(model, epoch, csvwriter):
    model.train()
    train_losses = list()
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_(True)  # set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x8
        loss = F.cross_entropy(output[0], target)  # the loss functions expects a batchSizex8 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss))
            csvwriter.writerow(["Train", epoch, batch_idx * len(data), len(trainloader.dataset),
                                100. * batch_idx / len(trainloader), loss])
    return train_losses


def test(model, epoch, csvwriter):
    model.eval()
    correct = 0
    test_accuracies = list()
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
        test_accuracies.append(correct / len(testloader.dataset))
    print('\nTest set: Epoch {} Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    csvwriter.writerow(["Test", epoch, correct, len(testloader.dataset),
                        100. * correct / len(testloader.dataset)])
    return test_accuracies


if __name__ == "__main__":

    # TODO Save results to a csv + plots
    # TODO Make inferences about the accuracies of the predicted classes
    dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outcsv_string = "output" + dt_string + ".csv"

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using cuda.\n')
    else:
        print('Using cpu.\n')
    model = ThreeVGCAudio1D()
    # model = ThreeVGCAudio2D()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    n_fold = 10
    # Run csv_extraction_script.py to get these files
    train_ = ['train' + str(i) + '.csv' for i in range(n_fold)]
    test_ = ['test' + str(i) + '.csv' for i in range(n_fold)]

    with open(outcsv_string) as outcsv:
        csvwriter = csv.writer(outcsv, delimiter=';',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(n_fold):
            # 16h of train set, 4h of test set
            # Look in csv_extraction_script to estimate how much the maximum values are
            train_csv_path = train_[i]
            test_csv_path = test_[i]
            train_set = AudioDataset(csv_path=train_csv_path, duration_per_category=16)
            test_set = AudioDataset(csv_path=test_csv_path, duration_per_category=4)
            print("Train set size: " + str(len(train_set)))
            print("Test set size: " + str(len(test_set)))

            kwargs = {'num_workers': 4,
                      'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

            trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)

            log_interval = 20
            for epoch in range(1, 41):
                if epoch == 21:
                    print("First round of training complete. Setting learn rate to 0.0001.")
                train(model, epoch, csvwriter)
                test(model, epoch, csvwriter)
                scheduler.step()
            del trainloader
            del testloader
            del train_set
            del test_set
