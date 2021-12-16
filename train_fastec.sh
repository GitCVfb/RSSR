##!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
fastec_dataset_type=Fastec
fastec_root_path_training_data=/home1/fanbin/fan/raw_data/faster/data_train/train/

log_dir=/home1/fanbin/fan/RSSR/deep_unroll_weights/
log_dir_pwc=/home1/fanbin/fan/RSSR/deep_unroll_weights/pretrain_pwc/

cd deep_unroll_net


python train_RSSR.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_training_data \
          --log_dir=$log_dir \
          --log_dir_pwc=$log_dir_pwc \
          --lamda_L1=10 \
          --lamda_L1_rs=10 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --crop_sz_H=448 \
          #--continue_train=True \
          #--start_epoch=51 \
          #--model_label=50 \