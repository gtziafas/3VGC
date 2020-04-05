import torch
import torchaudio
import pandas as pd
import numpy as np

class ThreeVGCAudio(Dataset):
    
    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0,len(csvData)):
            if csvData.iloc[i, 2] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 1])
                self.folders.append(csvData.iloc[i, 2])
                
        self.file_path = file_path
        self.folderList = folderList

        new_sr = sr/5
        channel = 1
        
    def __getitem__(self, index):
        #format the file path and load the file
        path = self.file_path + str(self.folders[index]) + "/audio/" + self.file_names[index]
        #load returns a tensor with the sound data and the sampling frequency (44.1kHz)
        #Original sound size ([2, 221184])
        sound, sr = torchaudio.load(path, out = None, normalization = True)
        #Downsample to ~8Khz , soundData size ([1, 44237]) , 
        transformed =  torchaudio.transforms.Resample(sr, new_sr)(sound[channel,:].view(1,-1))
        #swap dimensions
        soundData = transformed[0].unsqueeze(0)
        #pad the sound from 44237 to 32000 
        sound_padded = soundData[:, :32000]
        #sound_padded = sound_padded.permute(1,0)


        return sound_padded, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)


csv_path = 'trantas_data.csv'
file_path = 'data/'
#Category folders, renamed manually to 1,2
#category_list = ['Documentary = 1', 'vlog = 2 ', 'test = 3']

train_set = ThreeVGCAudio(csv_path, file_path, [1,2])
test_set = ThreeVGCAudio(csv_path, file_path, [3])
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)

