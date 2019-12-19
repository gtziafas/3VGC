import os
import shutil

import youtube_dl
import subprocess

"""Download a video in n second fragments"""
"""Requires that ffmpeg and ffprobe are installed!"""


def get_length(input_video):
    result = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet',
                                      '-of', 'csv=%s' % "p=0"])
    return result.decode("utf-8")


def download_videos(url: str, class_string: str, n_seconds: int = 30, split_videos=True, split_subs=True):
    # Create output dir
    output_directory = "out/" + class_string
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        confirmation = 'a'
        while confirmation != 'y' and confirmation != 'n':
            confirmation = input("Delete current contents of output folder? (y/n): ")
        if confirmation == 'y':
            for filename in os.listdir(output_directory):
                filepath = os.path.join(output_directory, filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)

    # Download the full video with subtitles
    ydl_opts = {
        'outtmpl': output_directory + '/%(id)s%(ext)s',
        'writesubtitles': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Lists of current downloaded files
    file_list = os.listdir(output_directory)
    video_list = list(filter(lambda x: x.endswith(".mp4") or x.endswith(".mkv"), file_list))
    subs_list = list(filter(lambda x: x.endswith(".vtt"), file_list))

    # Split videos
    if split_videos:
        for index, filename in enumerate(video_list):
            # Get video length
            length = int(float(get_length(output_directory + '/' + filename)))

            print("video ", index, " has length ", length, "and", int(length / n_seconds) - 1, " fragments")

            # Divide duration by n to get range size
            for fragment in range(0, int(length / n_seconds) - 1):
                start_seconds = fragment * n_seconds
                start_time = str(int(start_seconds / 60)) + ":" + str(start_seconds % 60)
                print(start_time)
                os.system('ffmpeg -ss ' + start_time + ' -i "' +
                          output_directory + '/' + filename +
                          '" -ss ' + start_time +
                          ' -i "' + output_directory + '/' + filename +
                          '" -t 0:' + str(n_seconds) + ' -map 0:v -map 1:a -c:v libx264 -c:a aac ' + \
                          output_directory + '/' + os.path.splitext(filename)[0] + str(fragment) + '.mkv')

            # Remove the file after it was split
            if os.path.exists(output_directory + '/' + filename):
                os.remove(output_directory + '/' + filename)

    # Split subtitles
    if split_subs:
        for index, filename in enumerate(subs_list):
            # Split subtitle files
            # TODO
            pass


if __name__ == "__main__":
    download_videos('https://www.youtube.com/watch?v=h2uDE8TuULs', "drama", 30, split_videos=False)
