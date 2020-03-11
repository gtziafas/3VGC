# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:42:00 2020

@author: battu
"""

import os
import random
from math import ceil
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print("\nTraining on GPU")
else:
    print("\nNo GPU, training on CPU")


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


def load_features_into_data(waveform, data, labels, file_number, file_name):
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
    label = file_name.split('_')[0]
    for i in range(3):
        labels[file_number][i] = label[i]
    return data, labels


# Reading the data
# TODO: make a data loader
train_directory = '/data/s4120310/train_audio_data'
test_directory = '/data/s4120310/test_audio_data'
data_directory = '/data/s4161947/sports/audio'


def get_data(train_dir=train_directory, test_dir=test_directory, data_dir=data_directory, shuffle_seed=None):
    # train_pathlist = Path(train_dir).glob('**/*')
    # test_pathlist = Path(test_dir).glob('**/*')

    # n_train = len(next(os.walk(train_dir)))
    # n_test = len(next(os.walk(test_dir)))

    pathlist = Path(data_dir).glob('**/*')

    data_size = len(next(os.walk(data_dir)))

    n_train = int(ceil(data_size * 8/10))
    n_test = data_size - n_train

    if shuffle_seed is not None:
        random.seed(shuffle_seed)

    random.shuffle(pathlist)

    train_pathlist = pathlist[:n_train]
    test_pathlist = pathlist[n_train:]

    print(len(train_pathlist))
    print(len(test_pathlist))

    exit(0)

    train_X = np.zeros(
        (n_train, 217, 174), dtype=np.float64
    )
    train_Y = np.zeroes(
        (n_train, 3), dtype=np.float64
    )
    test_X = np.zeros(
        (n_test, 217, 174), dtype=np.float64
    )
    test_Y = np.zeroes(
        (n_test, 3), dtype=np.float64
    )

    # Filling training data
    count = 0
    for path in train_pathlist:
        file = str(path)
        file_name = file.split('/')[-1]
        waveform, sr = librosa.load(file)
        train_X, train_Y = load_features_into_data(waveform, train_X, train_Y, count, file_name)
        count += 1

        # Filling testing data
    count = 0
    for path in test_pathlist:
        file = str(path)
        file_name = file.split('/')[-1]
        waveform, sr = librosa.load(file)
        test_X, test_Y = load_features_into_data(waveform, test_X, test_Y, count, file_name)
        count += 1
    torch_train_X = torch.from_numpy(train_X).type(torch.Tensor)
    torch_train_Y = torch.from_numpy(train_Y).type(torch.LongTensor)
    torch_test_X = torch.from_numpy(test_X).type(torch.Tensor)
    torch_test_Y = torch.from_numpy(test_Y).type(torch.LongTensor)
    return torch_train_X, torch_train_Y, torch_test_X, torch_test_Y


def run_model(n_epochs=400, batch_size=35, lr=0.001):
    train_X, train_Y, test_X, test_Y = get_data(train_directory, test_directory)

    print("Build LSTM RNN model ...")
    model = LSTM(
        input_dim=174, hidden_dim=256, batch_size=batch_size, output_dim=3, num_layers=2
    )
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_batches = int(train_X.shape[0] / batch_size)

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

            y_pred = model(X_local_minibatch)  # fwd the bass (forward pass)
            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()  # reeeeewind (backward pass)
            optimizer.step()  # parameter update

            train_running_loss += loss.detach().item()  # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)

        print(
            "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / num_batches, train_acc / num_batches)
        )
