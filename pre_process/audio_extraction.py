# extract audio from a video, or for all videos in directory
# usage:    python3 audio_extraction './path/to/videos/' 'webm' 'mp3' 192000
# last 3 args are optional with the default values shown
import os
import sys

from pathlib import Path


def extract_audio_from_video(file: str, bitrate: int = 192000, out_format: str = 'mp3') -> None:
    crumbs = file.split("/")
    write_name = "/".join(crumbs[0:5]) + "/audio/" + crumbs[-1].split(".")[0] + '.' + out_format
    # print(file)
    # print(write_name)
    os.system('ffmpeg -i ' + file + ' -f ' + out_format + ' -ab ' + str(bitrate) + ' -vn ' + write_name)


def do_entire_dir(data_dir: str, in_format: str = 'webm', out_format: str = 'mp3', bitrate: int = 192000) -> None:
    pathlist = Path(data_dir).glob('**/*.' + in_format)
    for path in pathlist:
        file = str(path)
        extract_audio_from_video(file=file, out_format=out_format, bitrate=bitrate)


def as_script(args) -> None:
    # parse arguements
    data_dir = args[1]
    in_format = 'webm' if len(args) < 3 else args[2]
    out_format = 'mp3' if len(args) < 4 else args[3]
    bitrate = 192000 if len(args) < 5 else eval(args[4])

    do_entire_dir(data_dir, in_format, out_format, bitrate)


if __name__ == "__main__":
    as_script(sys.argv)
