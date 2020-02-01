## USAGE: python3 subsample_videos.py ~/.../videos/ 15 mkv mkv
## script DEMANDS a dir named videos_subsampled to exist in the same lvl as ~/.../videos/
## to save all subsampled videos
import cv2
import os
import sys
import pickle
import numpy as np
from pathlib import Path


def parse_dir_names(path: str):
    # parse directory lvls until reaching filename and drop extension
    tokens = path.split('/')
    filename = tokens[-1].split('.')[0]
    dirname = tokens[-2]
    pre = '/'.join(tokens[0:-2])

    return filename, dirname, pre


def subsample_video(path: str, total_frames: int, out_format: str):
    # load video with openCV
    cap = cv2.VideoCapture(path)

    # get the filename, video directory name and pre-name path
    filename, dirname, pre = parse_dir_names(path)

    # total number of frames, video resolution and framerate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # buffer data
    buffered = np.empty((frame_count, height, width, 3), dtype='uint8')
    ret, f = True, 0
    while f < frame_count and ret:
        ret, buffered[f, :, :, :] = cap.read()
        f += 1
    cap.release()

    # init video saver oject
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    try:
        os.mkdir(pre + '/' + dirname + '_subsampled/')
    except:
        pass
    out_name = pre + '/' + dirname + '_subsampled/' + filename + '_subsampled.' + out_format
    out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

    # sub-sample data to desired total number of frames
    step = int(frame_count / total_frames)
    subsampled = buffered[0::step]

    print('Saving to: {} ....'.format(out_name))
    for _ in range(total_frames):
        out.write(subsampled[_])
    out.release()

    return subsampled


def do_entire_dir(args):
    # parse args
    data_dir = args[1]  # path to read video files
    total_frames = int(args[2])  # total number of frames that left after sub-sampling
    in_format = args[3]  # input video format (default: '.mkv')
    out_format = args[4]  # output video format (default: '.mkv')
    save_to_pickle = False if args[5] == None else args[5]

    # subsample all videos in data_dir
    pathlist = Path(data_dir).glob('**/*.' + in_format)
    total = []
    for path in pathlist:
        file = str(path)
        print('Subsampling video file: {} ....'.format(file))
        subd = subsample_video(file, total_frames, out_format)
        total.append(subd.transpose())

    # save data to a pickle file if requested
    if save_to_pickle:
        with open('videos_subsamples.pkl', 'wb') as handle:
            pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    do_entire_dir(sys.argv)
