#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_carla_video
mkdir -p experiments/results_demo_faster_video

cd deep_unroll_net


python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla_video \
            --data_dir='../demo/Carla' \
            --log_dir=../deep_unroll_weights/carla
            
python inference_demo_video.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_faster_video \
            --data_dir='../demo/Fastec' \
            --is_Fastec=1 \
            --log_dir=../deep_unroll_weights/fastec
