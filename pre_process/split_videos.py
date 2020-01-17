import csv
import sys
import time

import youtube_dl
import os 
import subprocess

from pathlib import Path

def get_length(input_video):
    result = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet',
                                      '-of', 'csv=%s' % ("p=0")])
    return eval(result.decode("utf-8"))

def split_video(file: str, dur: int, write_name: str, out_format: str) -> None:
    length = get_length(file)
    chunks = int(length / dur) 

    for i in range(chunks):
        start_time = i * dur 
        os.system('ffmpeg -strict experimental -i ' + file + ' -ss ' + str(start_time) +' -t ' + str(dur) + ' ' + write_name + '_' + str(i) + '.' + out_format)


def split_entire_dir(data_dir: str, duration: float, in_format: str = 'mp4', out_format: str = 'mp4') -> None:
    duration = eval(duration)
    pathlist = Path(data_dir).glob('**/*.' + in_format)

    crumbs = data_dir.split("/")
    out_dir = "/".join(crumbs[0:5]) + "/video/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in pathlist:
        file = str(path)
        #print(file)
        # break
        crumbs = file.split("/")
        out_name = "/".join(crumbs[0:5]) + "/video/" + crumbs[-1].split(".")[0] + "_parsed"
        #print(out_name)
        # break
        split_video(file, duration, out_name, out_format)

if __name__ == "__main__":
    split_entire_dir(*sys.argv[1:])