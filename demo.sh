#!/bin/bash

# create an empty folder for experimental results
mkdir -p experiments/results_demo_fastec
mkdir -p experiments/results_demo_carla

cd deep_unroll_net

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_carla \
            --data_dir='../demo/Carla' \
            --is_Fastec=0 \
            --log_dir=../deep_unroll_weights/carla

python inference_demo.py \
            --model_label='pre' \
            --results_dir=../experiments/results_demo_fastec \
            --data_dir='../demo/Fastec' \
            --is_Fastec=1 \
            --log_dir=../deep_unroll_weights/fastec
