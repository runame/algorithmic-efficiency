#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8G

# NOTE1 Before being able to run this, you need to carry out the data preparation:
# 1) Create a folder `data`: `mkdir data`
# 2) Link ImageNet:
#    - `mkdir data/imagenet`
#    - `ln -s /datasets/imagenet/train data/imagenet/train`
#    - `ln -s /datasets/imagenet/val data/imagenet/val`
# 3) NOTE The first time you run an ImageNet workload, it will download another
#    data set (ImageNetV2) into `data/imagenet/imagenetv2` (size < 2.0 GiB).
#    Make sure you call a workload for the first time using ONLY ONE GPU, otherwise
#    all GPUs will simultaneously try to download and extract the data set, which will
#    cause errors.
#
# NOTE2 To use more GPUs change `--gres=gpu:X` above, and `nproc_per_node=X` to a larger number
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=2 submission_runner.py \
    --framework=pytorch \
    --data_dir=data/imagenet/ \
    --experiment_dir=experiments/ \
    --experiment_name=imagenet_vit_first_try \
    --workload=imagenet_vit \
    --submission_path=submissions/submission_folder/external_tuning/sirfshampoo/submission.py \
    --tuning_search_space=submissions/submission_folder/external_tuning/sirfshampoo/tuning_search_space.json \
    --overwrite=True  \
    --save_checkpoints=False #\
#    --use_wandb \
