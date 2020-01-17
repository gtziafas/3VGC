1) Create 2 category folders in `/data/sxxxxxxx/`

`cartoon`

`|raw
|video
|video_subsampled
|audio
|text`

2) Put all your urls in `playlists.csv`

3) `git clone` the repo somewhere in the `/home` folder

4) From the git repo root, open the Python console:

`> from pre_process.youtube_downloader import *`
`> download_dataset(textfile="playlists.csv")`
5) Convert all downloaded videos to `.mp4` using `ffmpeg`
`for i in *.webm; do ffmpeg -i "$i" "${i%.*}.mp4"; done`

6) Open up the console like in step 4

`from pre_process.youtube_downloader import *`
`> split_entire_dir(data_dir="/data/sxxxxxxx/cartoon/raw/", duration=5, in_format="mp4", out_format="mp4")`

7) Subsample videos to keep only 4 frames per sec. Run `python pre_process/subsample_videos.py * 20 'mp4' 'mp4'`

8) Run the audio extraction `python pre_process/audio_extraction.py /data/sxxxxxxx/cartoon/video/ "mp4" "mp3"`

9) Run the subtitle extraction  for 5 second chunks `python pre_process/subs_parser.py /data/sxxxxxxx/cartoon/raw/ 5`