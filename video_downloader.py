import os
import youtube_dl

"""Download a video in n second fragments"""


def download_video(url: str, class_string: str, n: int):
    # Create output dir
    output_directory = "out/" + class_string
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Download the video in n second fragments
    for fragment in range(0, 2):
        start_time = fragment * n

        ydl_opts = {
            'outtmpl': output_directory + '/%(id)' + str(n) + 's%(ext)s',
            'writesubtitles': True
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    # Sample


download_video('https://www.youtube.com/watch?v=QjL7D33xpS4', "drama", 5)
