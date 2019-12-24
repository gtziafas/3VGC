import youtube_dl
import os 
import subprocess

out_dir = './Desktop/youtube_dl_ds/'

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

def split_video(file: str, dur: int) -> None:
    length = get_length(file)
    chunks = int(length / dur) 
    
    ext = file.split('.')[2]
    filename = (file.split('.')[1]).split('/')[-1]

    for i in range(chunks):
        start_time = i * dur 
        os.system('ffmpeg -i ' + file + ' -ss ' + str(start_time) +' -t ' + str(dur) + ' ' + filename + '_' + str(i) + '.' + ext)

split_video('./out/drama/h2uDE8TuULsmp4.mkv', 5)
