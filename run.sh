#!/bin/bash
#SBATCH --job-name="bubbleshort"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:1  


module purge
module load Python FFmpeg 

python ~/3VGC/pre_process/split_videos.py /data/s3913171/3VGC/cartoon/raw/ 5 webm mp4 
