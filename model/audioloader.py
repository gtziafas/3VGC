import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(self, csv_path, duration_per_category, new_freq=8000):
        """
        Initialize an audio dataset
        :param csv_path: path of csv file containing paths, generated with csv_extraction_script.py
        :param duration_per_category: number of hours to load per category
        :param new_freq: frequency to sample at
        """
        csv_data = pd.read_csv(csv_path)
        self.file_path = file_path
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # Assume each sample is 5 seconds, there are 720 samples per hour
        self.samples_per_category = duration_per_category * 720
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csv_data)):
            if i > self.samples_per_category:
                break
            else:
                self.file_names.append(csv_data.iloc[i, 0])
                self.labels.append(csv_data.iloc[i, 1])

        self.channel = 1

        self.new_freq = new_freq

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz)
        # Original sound size ([2, 221184])
        sound, sr = torchaudio.load(path, out=None, normalization=True)
        # Downsample to ~8Khz , sound_data size ([1, 44237]) ,
        transformed = torchaudio.transforms.Resample(sr, self.new_freq)(sound[self.channel, :].view(1, -1))
        # swap dimensions
        sound_data = transformed[0].unsqueeze(0)
        # pad the sound from 44237 to 32000
        sound_padded = sound_data[:, :40000]
        # sound_padded = sound_padded.permute(1,0)

        return sound_padded, self.labels[index]

    def __len__(self):
        return len(self.file_names)


if __name__ == "__main__":
    csv_path = 'trantas_data.csv'
    file_path = 'data/'
    # Category folders, renamed manually to 1,2
    # category_list = ['Documentary = 1', 'vlog = 2 ', 'test = 3']

    train_set = AudioDataset(csv_path, file_path, [1, 2])
    test_set = AudioDataset(csv_path, file_path, [3])
    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(test_set)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using {}\n'.format(device))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)
