import os
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

#I'm extracting for music and drama respectively, so i use
labels = [
    '00001000',
    '00100000'
]

# I give the paths in the similar order, music first and dramas next, make sure to have the / at the end
paths = [
    '/data/s4120310/music/audio/',
    '/data/s4120310/dramas/audio/'
]
#Extract your two categories for now, we can combine the csv files later on
data = pd.DataFrame(columns=['Filename','Label'])

for i in range(len(labels)):
    path = paths[i]
    k =os.listdir(path)
    for j in k:
      data.loc[data.shape[0]] =  [path+j,labels[i]]

data.to_csv('surender_data.csv',index=False)
    

