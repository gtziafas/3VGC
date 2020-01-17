import sys
import time
import datetime

from collections import defaultdict
from pathlib import Path 
from typing import List, Tuple


def remove_indexing(lines: List[str]) -> List[str]:
    return list(filter(lambda l: not l.split('\n')[0].isdigit(), lines))


def parse_file(file: str, manually_downloaded_flag: bool = False) -> List[Tuple[float, str]]:
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = remove_indexing(lines)

    def _parse_file(lines: List[str]) -> List[Tuple[float, str]]:
        tfs, subs = defaultdict(), defaultdict()
        for i, l in enumerate(lines): 
            if '-->' in l:
                flag_char = '.' if not manually_downloaded_flag else ','
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

    return _parse_file(lines)


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
    manually_downloaded_flag = bool(args[3])            # set to positive integer if subs are manually downloaded
    in_format = 'vtt' 

    pathlist = Path(data_dir).glob('**/*.' + in_format) 
    for path in pathlist:
        file = str(path)
        out_name = file.split('.')[0] + '_parsed.' + in_format

        with open(out_name, 'a') as f:
            file_data = parse_file(file, manually_downloaded_flag)
            parsed = split_to_samples(file_data, duration)
            for p in parsed:
                f.write('- ' + p + '\n')    


if __name__ == "__main__":
    do_entire_dir(sys.argv)