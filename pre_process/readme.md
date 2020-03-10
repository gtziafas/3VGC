# Usage

0) In Peregrine:
```module purge```,
```module load Python PyTorch FFmpeg OpenCV```,
```pip install youtube_dl --user```

1) Create 2 category folders in `/data/sxxxxxxx/`

`|raw
|video
|video_subsampled
|audio
|text`

2) Put all your urls in `playlists.csv`

3) `git clone` the repo somewhere in the `/home` folder

4) From the git repo root, open the Python console:

`> from pre_process.youtube_downloader import *`

`> download_dataset(textfile="playlists.csv", out_dir = "/data/sxxxxxx/cartoon/raw")`,

5) Convert all downloaded videos to `.mp4` using ffmpeg:
 `for i in *.webm; do ffmpeg -i "$i" "${i%.*}.mp4"; done`

6) Run the following script to split all videos in your data folders into 5 second duration samples:
`python pre_process/split_videos.py /data/sxxxxxx/cartoon/video/raw 5 mp4 mp4`

7) Subsample videos to keep only 4 frames per sec. Run:
`python pre_process/subsample_videos.py /data/sxxxxxx/cartoon/video 20 'mp4' 'mp4' False`

8) Run the audio extraction `python pre_process/audio_extraction.py /data/sxxxxxxx/cartoon/video/ "mp4" "mp3"`

9) Run the subtitle extraction  for 5 second chunks `python pre_process/subs_parser.py /data/sxxxxxxx/cartoon/raw/ 5 False`
