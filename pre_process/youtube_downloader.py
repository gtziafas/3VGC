import csv
import time

import youtube_dl
import os
import subprocess

from pathlib import Path


def get_length(input_video):
    result = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet',
                                      '-of', 'csv=%s' % ("p=0")])
    return eval(result.decode("utf-8"))


def download_video_from_url(url: str, out_dir: str) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    options = {
        'format': 'worst',
        'outtmpl': out_dir + '/%(id)s.%(ext)s',
        'writesubtitles': True,
        'writeautomaticsub': True
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([url])


def split_video(file: str, dur: int, write_name: str, out_format: str) -> None:
    length = get_length(file)
    chunks = int(length / dur)

    for i in range(chunks):
        start_time = i * dur
        os.system('ffmpeg -i ' + file + ' -ss ' + str(start_time) + ' -t ' + str(dur) + ' ' + write_name + '_' + str(
            i) + '.' + out_format)


def split_entire_dir(data_dir: str, duration: float, in_format: str, out_format: str) -> None:
    pathlist = Path(data_dir).glob('**/*.' + in_format)

    crumbs = data_dir.split("/")
    out_dir = "/".join([crumbs[0], crumbs[1]]) + "/video/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for path in pathlist:
        file = str(path)
        print(file)
        # break
        crumbs = file.split("/")
        out_name = "/".join([crumbs[0], crumbs[1]]) + "/video/" + crumbs[-1].split(".")[0] + "_parsed"
        print(out_name)
        # break
        split_video(file, duration, out_name, out_format)


def download_dataset(textfile: str, n_seconds=30):
    label = "vlog"
    playlist_durations = {}
    with open(textfile) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[0] == "l":
                label = row[1]
            elif row[0] == "p":
                time_str = row[1]
                time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], time_str.split(":")))
                if label in playlist_durations:
                    playlist_durations[label] += time_seconds
                else:
                    playlist_durations[label] = time_seconds
                # Uncomment this to activate downloading
                download_video_from_url(row[2], "out/" + label + "/raw/")
    for label in playlist_durations:
        print(label, time.strftime('%H:%M:%S', time.gmtime(playlist_durations[label])))
