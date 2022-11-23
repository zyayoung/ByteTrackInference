#!/usr/bin/env bash

set -x
set -p

# for i in 0 1 2 3 4 5 6 7;
# do
# RLAUNCH_REPLICA=$i RLAUNCH_REPLICA_TOTAL=8 CUDA_VISIBLE_DEVICES=$i python tools/detect.py -f exps/example/mot/yolox_x_mix_det.py -c YOLOX_outputs/yolox_x_mix_hie20_ch/latest_ckpt.pth.tar --fp16 --fuse --pathname "/data/datasets/HIE20/*/*/img1/*.jpg" &
# done

# wait

python tools/detect.py -f exps/example/mot/yolox_x_mix_det.py -c ocsort_dance_model.pth.tar --fp16 --fuse --pathname "$1"
