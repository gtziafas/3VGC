import os
import glob
from pathlib import Path
import shutil
import random 

lst=[]
for i in range(1,9):
    data_dir = 'data/' + str(i) + '/audio/'
    pathlist = Path(data_dir).glob('**/*.' + 'wav')
    count=0
    file=[]
    new=[]
    for path in pathlist:
        file = str(path)
        new.append(file)
        crumbs = os.path.split(path)
        label = crumbs[-1].split("_")[-2]
        count += 1
    x = int((count * 20)/100)
    new_sample = random.sample(new, x)
    print(len(new_sample))
    for s in new_sample:
        shutil.move(s, 'data/9/audio')
    lst.append(x)
