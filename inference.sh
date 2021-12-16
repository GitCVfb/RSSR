#--load_1st_GS=0 ==> Corresponding to the middle scanline of second RS frame
#--load_1st_GS=1 ==> Corresponding to the first scanline of second RS frame

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_test_data=/home1/fanbin/fan/raw_data/carla/data_test/test/

fastec_dataset_type=Fastec
fastec_root_path_test_data=/home1/fanbin/fan/raw_data/faster/data_test/test/

model_dir_carla=../deep_unroll_weights/carla/
model_dir_fastec=../deep_unroll_weights/fastec/

results_dir=/home1/fanbin/fan/RSSR/deep_unroll_results/

cd deep_unroll_net

python inference.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_test_data \
          --log_dir=$model_dir_carla \
          --results_dir=$results_dir \
          --compute_metrics \
          --model_label=pre \
          --load_1st_GS=1

#python inference.py \
#          --dataset_type=$fastec_dataset_type \
#          --dataset_root_dir=$fastec_root_path_test_data \
#          --log_dir=$model_dir_fastec \
#          --results_dir=$results_dir \
#          --compute_metrics \
#          --model_label=pre \
#          --load_1st_GS=1


