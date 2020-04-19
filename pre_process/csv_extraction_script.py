import os
import random as rn

import pandas as pd

'''
Use This convention in the labels for your categories
vlog :             00000001
documentary:       00000010
sports:            00000100
music:             00001000
esports:           00010000
drama:             00100000
tv-show:           01000000
cartoon:           10000000

'''

# I'm extracting for music and drama respectively, so i use
labels = [
    '00001000',
    '00100000',
    '00000100',
    '01000000'
]

# I give the paths in the similar order, music first and dramas next, make sure to have the / at the end
paths = [
    '/data/s4120310/music/audio/',
    '/data/s4120310/dramas/audio/'
    '/data/s4161947/sports/audio/',
    '/data/s4161947/tv/audio/'
]


def get_train_and_test_csv(seed=10, train_ratio=0.8):
    """
    Shuffle the dataset, then split it into train and test, then generate a csv file for each
    :param seed: seed for the random shuffling; set to None for non reproducible results
    :param train_ratio: ratio of dataset that should be used for training
    :return: names of the csv files
    """
    # Extract four categories for now, we can combine the csv files later on
    train_files = pd.DataFrame(columns=['Filename', 'Label'])
    test_files = pd.DataFrame(columns=['Filename', 'Label'])

    for i in range(len(labels)):
        path = paths[i]
        file_name_list = os.listdir(path)
        # Randomly shuffle file name list in place
        rn.Random(seed).shuffle(file_name_list)
        file_counter = 0
        # Train frame is train_ratio of the dataset
        while file_counter < train_ratio * len(file_name_list):
            train_files.loc[train_files.shape[0]] = [path + file_name_list[file_counter], labels[i]]
            file_counter += 1
        # Write the rest to the test frame
        while file_counter < len(file_name_list):
            test_files.loc[test_files.shape[0]] = [path + file_name_list[file_counter], labels[i]]
            file_counter += 1

    train_files.to_csv('train_file_paths.csv', index=False)
    test_files.to_csv('test_file_paths.csv', index=False)

    return 'train_file_paths.csv', 'test_file_paths.csv'
