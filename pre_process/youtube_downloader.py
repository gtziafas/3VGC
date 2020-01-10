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
        'format'            :   'worst',
        'outtmpl'           :   out_dir + '/%(id)s.%(ext)s',
        'writesubtitles'    :   True
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([url])


def split_video(file: str, dur: int, write_name: str, out_format: str) -> None:
    length = get_length(file)
    chunks = int(length / dur) 

    for i in range(chunks):
        start_time = i * dur 
        os.system('ffmpeg -i ' + file + ' -ss ' + str(start_time) +' -t ' + str(dur) + ' ' + write_name + '_' + str(i) + '.' + out_format)


def split_entire_dir(data_dir: str, duration: float, in_format: str, out_format: str) -> None:
    pathlist = Path(data_dir).glob('**/*.' + in_format) 
    for path in pathlist:
        file = str(path)
        #out_name = file.split('.')[0] + '_parsed.' + in_format
        print(file)
        split_video(file, duration, 'dfsdd', out_format)

