# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:42:00 2020

@author: battu
"""

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle
import threading

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\nTraining on GPU")
    device = torch.device("cuda:0")
else:
    print("\nNo GPU, training on CPU")
    device = torch.device("cpu")

import warnings

warnings.filterwarnings("ignore")


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=3, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input):
        # lstm step => then ONLY take the sequence's final timetep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
                torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()


def load_features_into_data(waveform, data, labels, file_number, file_name, label):
    mfcc = librosa.feature.mfcc(waveform, sr=44100, n_mfcc=15)
    mel = librosa.feature.melspectrogram(waveform, sr=44100)
    cens = librosa.feature.chroma_cens(waveform, sr=44100, n_chroma=15)
    ton = librosa.feature.tonnetz(waveform, sr=44100, chroma=cens)
    cqt = librosa.feature.chroma_cqt(waveform, sr=44100, n_chroma=5)
    stft = librosa.feature.chroma_stft(waveform, sr=44100, n_chroma=5)
    data[file_number, :, 0:15] = mfcc.T
    data[file_number, :, 15:143] = mel.T
    data[file_number, :, 143:158] = cens.T
    data[file_number, :, 158:164] = ton.T
    data[file_number, :, 164:169] = cqt.T
    data[file_number, :, 169:174] = stft.T
    label = label
    for i in range(8):
        labels[file_number][i] = int(label[i])
    return data, labels


# Reading the data
train_directory = 'train_file_paths.csv'
test_directory = 'test_file_paths.csv'


def get_data(train_csv=train_directory, test_csv=test_directory):
    print("Loading Data!....")
    # train_pathlist = Path(train_dir).glob('**/*')
    # test_pathlist = Path(test_dir).glob('**/*')

    c = torch.zeros(4, 3)
    c = c.to(device)

    classes = 6
    n_train = 60000
    n_test = 18000

    train_sample_per_class = int(n_train / classes)
    test_sample_per_class = int(n_test / classes)

    train_X = np.zeros(
        (n_train, 217, 174), dtype=np.float64
    )
    train_Y = np.zeros(
        (n_train, 8), dtype=np.float64
    )
    test_X = np.zeros(
        (n_test, 217, 174), dtype=np.float64
    )
    test_Y = np.zeros(
        (n_test, 8), dtype=np.float64
    )

    # Filling training data
    def train_thread():
        nonlocal train_X, train_Y, n_train, train_sample_per_class
        print("Training_Thread")
        a = pd.read_csv(train_csv)
        a = a.groupby('Label').apply(lambda x: x.sample(10000)).reset_index(drop=True)
        count = 0
        for i in range(n_train):
            # print(i)
            file_name = a.loc[i][0]
            label = format(a.loc[i][1], '#08d')
            # file_name = file.split('/')[-1]

            try:
                waveform, sr = librosa.load(file_name)
                train_X, train_Y = load_features_into_data(waveform, train_X, train_Y, count, file_name, label)
                count += 1
            except Exception as e:
                print("Missed data point at", count, e)
        print("Training Count:", count)
        train_X = train_X[:-(n_train - count - 1)]
        train_Y = train_Y[:-(n_train - count - 1)]

    # Filling testing data
    def test_thread():
        nonlocal test_X, test_Y, n_test, test_sample_per_class
        print("Testing_Thread")
        b = pd.read_csv(test_csv)
        b = b.groupby('Label').apply(lambda x: x.sample(3000)).reset_index(drop=True)
        count = 0
        for i in range(n_test):
            # print(i)
            file_name = b.loc[i][0]
            label = format(b.loc[i][1], '#08d')
            # file_name = file.split('/')[-1]

            try:
                waveform, sr = librosa.load(file_name)
                test_X, test_Y = load_features_into_data(waveform, test_X, test_Y, count, file_name, label)
                count += 1
            except Exception as e:
                print("Missed data point at", count, e)
        print("Testing Count:", count)
        test_X = test_X[:-(n_test - count - 1)]
        test_Y = test_Y[:-(n_test - count - 1)]

    t1 = threading.Thread(target=train_thread)
    t2 = threading.Thread(target=test_thread)
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    print("Data Loading done!, Setting them up!")
    torch_train_X = torch.from_numpy(train_X).type(torch.Tensor)
    torch_train_Y = torch.from_numpy(train_Y).type(torch.LongTensor)
    torch_test_X = torch.from_numpy(test_X).type(torch.Tensor)
    torch_test_Y = torch.from_numpy(test_Y).type(torch.LongTensor)
    return torch_train_X, torch_train_Y, torch_test_X, torch_test_Y


def run_data():
    train_X, train_Y, test_X, test_Y = get_data(train_directory, test_directory)

    for i in range(60):
        if i == 59:
            torch.save(train_X[i * 1000:].clone(), '/data/s4120310/tensors/train_x_' + str(i) + '.pt')
            torch.save(train_Y[i * 1000:].clone(), '/data/s4120310/tensors/train_y_' + str(i) + '.pt')
            break
        torch.save(train_X[i * 1000:(i + 1) * 1000].clone(), '/data/s4120310/tensors/train_x_' + str(i) + '.pt')
        torch.save(train_Y[i * 1000:(i + 1) * 1000].clone(), '/data/s4120310/tensors/train_y_' + str(i) + '.pt')

    for i in range(18):
        if i == 17:
            torch.save(test_X[i * 1000:].clone(), '/data/s4120310/tensors/test_x_' + str(i) + '.pt')
            torch.save(test_Y[i * 1000:].clone(), '/data/s4120310/tensors/test_y_' + str(i) + '.pt')

        torch.save(test_X[i * 1000:(i + 1) * 1000].clone(), '/data/s4120310/tensors/test_x_' + str(i) + '.pt')
        torch.save(test_Y[i * 1000:(i + 1) * 1000].clone(), '/data/s4120310/tensors/test_y_' + str(i) + '.pt')


def run_model(n_epochs=100, batch_size=20, lr=0.0001, load_model=False):
    print("Reloading Data...")

    train_X = torch.load('/data/s4120310/tensors/train_x_0.pt').to(device)
    train_Y = torch.load('/data/s4120310/tensors/train_y_0.pt').to(device)
    test_X = torch.load('/data/s4120310/tensors/test_x_0.pt').to(device)
    test_Y = torch.load('/data/s4120310/tensors/test_y_0.pt').to(device)
    for i in range(1, 60):
        train_X = torch.cat([train_X, torch.load('/data/s4120310/tensors/train_x_' + str(i) + '.pt')], 0)
        train_Y = torch.cat([train_Y, torch.load('/data/s4120310/tensors/train_y_' + str(i) + '.pt')], 0)
    for i in range(1, 18):
        test_X = torch.cat([test_X, torch.load('/data/s4120310/tensors/test_x_' + str(i) + '.pt')], 0)
        test_Y = torch.cat([test_Y, torch.load('/data/s4120310/tensors/test_y_' + str(i) + '.pt')], 0)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    print("Build LSTM RNN model ...")

    model = LSTM(
        input_dim=174, hidden_dim=256, batch_size=batch_size, output_dim=8, num_layers=2
    )

    if load_model:
        model.load_state_dict(torch.load('BestModel.mdl'))
    try:
        model.cuda()
    except Exception as e:
        print(e)
        pass
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_batches = int(train_X.shape[0] / batch_size)
    num_dev_batches = int(test_X.shape[0] / batch_size)
    print(train_X.shape, test_X.shape)
    val_loss_list, val_accuracy_list, epoch_list = [], [], []

    print("Training ...")
    for epoch in range(n_epochs):

        train_running_loss, train_acc = 0.0, 0.0

        model.hidden = model.init_hidden()
        for i in range(num_batches):
            model.zero_grad()

            X_local_minibatch, y_local_minibatch = (
                train_X[i * batch_size: (i + 1) * batch_size, ],
                train_Y[i * batch_size: (i + 1) * batch_size, ],
            )

            X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

            y_local_minibatch = torch.max(y_local_minibatch, 1)[1]
    
            y_pred = model(X_local_minibatch)                # fwd the bass (forward pass)
            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()                                  # reeeeewind (backward pass)
            optimizer.step()                                 # parameter update
    
            train_running_loss += loss.detach().item()       # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)

        print(
            "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / num_batches, train_acc / num_batches)
        )
        if n_epochs % 10 == 0:
            torch.save(model.state_dict(), 'BestModel.mdl')

        print("Validation ...")
        if epoch % 10 == 0:
            val_running_loss, val_acc = 0.0, 0.0

            # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
            with torch.no_grad():
                model.eval()

                model.hidden = model.init_hidden()
                for i in range(num_dev_batches):
                    X_local_validation_minibatch, y_local_validation_minibatch = (
                        test_X[i * batch_size: (i + 1) * batch_size, ],
                        test_Y[i * batch_size: (i + 1) * batch_size, ],
                    )
                    X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                    y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                    y_pred = model(X_local_minibatch)
                    val_loss = loss_function(y_pred, y_local_minibatch)

                    val_running_loss += (
                        val_loss.detach().item()
                    )  # unpacks the tensor into a scalar value
                    val_acc += model.get_accuracy(y_pred, y_local_minibatch)

                model.train()  # reset to train mode after iterationg through validation data
                print(
                    "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                    % (
                        epoch,
                        train_running_loss / num_batches,
                        train_acc / num_batches,
                        val_running_loss / num_dev_batches,
                        val_acc / num_dev_batches,
                    )
                )

            epoch_list.append(epoch)
            val_accuracy_list.append(val_acc / num_dev_batches)
            val_loss_list.append(val_running_loss / num_dev_batches)

    results = [epoch_list, val_accuracy_list, val_loss_list]
    f = open('/home/s4120310/results.pkl', 'wb')
    pickle.dump(results, f)
    f.close()


if __name__ == "__main__":
    # run_data() # Use with CPU
    run_model()  # Use with GPU
