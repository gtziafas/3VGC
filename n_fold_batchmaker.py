# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:47:44 2020

@author: battu
"""

import pandas as pd

samples_per_labels = 1000
tests_per_labels = 500
n_fold = 10


a = pd.read_csv("train_file_paths.csv")
b = pd.read_csv("test_file_paths.csv")



for i in range(n_fold):
    train_df = a.groupby('Label').apply(lambda x: x.sample(samples_per_labels)).reset_index(drop=False)
    test_df = b.groupby('Label').apply(lambda x: x.sample(tests_per_labels)).reset_index(drop=False)
    train_df.to_csv('train'+str(i)+'.csv')
    test_df.to_csv('test'+str(i)+'.csv')
    


