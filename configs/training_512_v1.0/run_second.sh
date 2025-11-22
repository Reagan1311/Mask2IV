# args
name=exp1
size=512

config_file_hoi4d=configs/training_${size}_v1.0/config_baseline_obj.yaml
config_file_bdv2=configs/training_${size}_v1.0/config_baseline_bdv2_maskcat.yaml

if [ "$1" = "hoi4d" ]; then
    config_file=$config_file_hoi4d
elif [ "$1" = "bdv2" ]; then
    config_file=$config_file_bdv2
else
    echo "Invalid argument: $1"
    exit 1
fi

# save_root="<YOUR_SAVE_ROOT_DIR>", for logs, checkpoints, tensorboard record, etc.
save_root="/workspace/exp_outputs/comparison/$1"

# run in multiple gpus
HOST_GPU_NUM=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1

## run in single gpu
# export LOCAL_RANK=0
# export RANK=0
# export WORLD_SIZE=0
# CUDA_VISIBLE_DEVICES=0 python ./main/trainer.py \
# --base $config_file \
# --train \
# --name $name \
# --logdir $save_root \
# --devices 1 \
# lightning.trainer.num_nodes=1

## validation in single gpu
# checkpoint=""
# first_mask="" # path for the first-stage mask results
# export LOCAL_RANK=0
# export RANK=0
# export WORLD_SIZE=0
# CUDA_VISIBLE_DEVICES=0 python ./main/trainer.py \
# --base $config_file \
# --val \
# --name $name \
# --logdir $save_root \
# --checkpoint $checkpoint \
# --devices 1 \
# --to_local \
# --to_img \
# --first_mask $first_mask \
# lightning.trainer.num_nodes=1
# # --val_save_skip \
