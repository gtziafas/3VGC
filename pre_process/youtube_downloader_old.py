import csv
import time

import youtube_dl
import os
import subprocess


def get_length(input_video):
    result = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet',
                                      '-of', 'csv=%s' % ("p=0")])
    return eval(result.decode("utf-8"))


def download_video_from_url(url: str, out_dir: str) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    options = {
        'format'            :   'worst',
        'outtmpl'           :   out_dir + '/%(id)s.%(ext)s',
        'writesubtitles'    :   True,
        'writeautomaticsub' :   True,
	'ignoreerrors'      :   True
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([url])


def download_dataset(textfile:str, out_dir:str, n_seconds=30):
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
                download_video_from_url(row[2], out_dir + label + "/raw/")
        for label in playlist_durations:
            m, s = divmod(playlist_durations[label], 60)
            h, m = divmod(m, 60)
            print(label, f'{h:d}:{m:02d}:{s:02d}')
