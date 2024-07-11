#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8G

# NOTE1 Before being able to run this, you need to carry out the data preparation:
# 1) Create a folder `data`: `mkdir data`
# 2) Download the data set: `python3 datasets/dataset_setup.py --data_dir data/ --ogbg1`
#
# NOTE2 To use more GPUs change `--gres=gpu:X` above, and `nproc_per_node=X` to a larger number
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=2 submission_runner.py \
    --framework=pytorch \
    --data_dir=data/ \
    --experiment_dir=experiments/ \
    --experiment_name=ogbg_first_try \
    --workload=ogbg \
    --submission_path=submissions/submission_folder/external_tuning/sirfshampoo/submission.py \
    --tuning_search_space=submissions/submission_folder/external_tuning/sirfshampoo/tuning_search_space.json \
    --overwrite=True  \
    --save_checkpoints=False #\
#    --use_wandb \
