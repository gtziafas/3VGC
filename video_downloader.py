import os
import youtube_dl
import subprocess

"""Download a video in n second fragments"""
"""Requires that ffmpeg and ffprobe are installed!"""


def get_length(input_video):
    result = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet',
                                      '-of', 'csv=%s' % ("p=0")])
    return result.decode("utf-8")


def download_video(url: str, class_string: str, n_seconds: int = 30):
    # Create output dir
    output_directory = "out/" + class_string
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Download the video in n second fragments
    ydl_opts = {
        'outtmpl': output_directory + '/%(id)s%(ext)s',
        'writesubtitles': True,
        'subtitlesformat': 'srt'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # Sample
    # For each downloaded file
    for index, filename in enumerate(os.listdir(output_directory)):
        print(index)

        # Only look at video files
        if filename.endswith(".mp4") or filename.endswith(".mkv"):

            length = int(float(get_length(output_directory + '/' + filename)))

            print(length, int(length / n_seconds) - 1)

            # Divide duration by n to get range size
            for fragment in range(0, int(length / n_seconds) - 1):
                start_seconds = fragment * n_seconds
                start_time = str(int(start_seconds/60)) + ":" + str(start_seconds % 60)
                print(start_time)
                os.system('ffmpeg -ss ' + start_time + ' -i "' +
                          output_directory + '/' + filename +
                          '" -ss ' + start_time +
                          ' -i "' + output_directory + '/' + filename +
                          '" -t 0:' + str(n_seconds) + ' -map 0:v -map 1:a -c:v libx264 -c:a aac ' + \
                          output_directory + '/' + os.path.splitext(filename)[0] + str(fragment) + '.mkv')

        if os.path.exists(output_directory + '/' + filename):
            os.remove(output_directory + '/' + filename)


if __name__ == "__main__":
    download_video('https://www.youtube.com/watch?v=h2uDE8TuULs', "drama", 30)
