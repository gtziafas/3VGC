import sys
import time
import datetime

from collections import defaultdict
from pathlib import Path 
from typing import List, Tuple

def parse_file(file: str) -> List[Tuple[float, str]]:
    with open(file, 'r') as f:
        lines = f.readlines()

    tfs, subs = defaultdict(), defaultdict()
    for i, l in enumerate(lines): 
         if '-->' in l: 
             tfs[i] = time.strptime(':'.join(l.split(':')[0:3]).split(' ')[0].split('.')[0], '%H:%M:%S')
             tfs[i] = datetime.timedelta(hours=tfs[i].tm_hour, minutes=tfs[i].tm_min, seconds=tfs[i].tm_sec).total_seconds()
         elif l[0] != '\n': 
             subs[i] = l.split('\n')[0] 
             if lines[i+1] != '\n': 
                 subs[i] = subs[i] + ' ' + lines[i+1].split('\n')[0]  
                 del lines[i+1] 
    tfs = list(tfs.values())
    subs = list(subs.values())[2:]

    return list(zip(tfs, subs))

def split_to_samples(data: List[Tuple[float, str]], duration: float) -> List[str]:
    res = [] 
    prev, idx, i = 0, 0, 0 
    for i, (tf, sub) in enumerate(data):
        if tf // duration > prev:
            res.append(' '.join([l[1] for l in data[idx:i]]))
            prev += 1 
            idx = i 
    res.append(' '.join([l[1] for l in data[idx:i+1]]))

    return res 

def do_entire_dir(args):
    # parse args
    data_dir = args[1]                                  # path to read subtitle files
    duration = float(args[2])                           # total duration of seconds for each text sample 
    in_format = 'vtt' 

    pathlist = Path(data_dir).glob('**/*.' + in_format) 
    out_name = '.'.join(data_dir.split('.')[0:2]) + '_parsed.' + in_format

    with open(out_name, 'a') as f:
        for path in pathlist:
            file = str(path)
            file_data = parse_file(file)
            parsed = split_to_samples(file_data, duration)
            for p in parsed:
                f.write('- ' + p + '\n')    


if __name__ == "__main__":
    do_entire_dir(sys.argv)