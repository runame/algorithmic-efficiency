#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=m4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-2

WORKLOADS_AND_DATA_PATHS=(
    "fastmri fastmri"
    "imagenet_vit imagenet"
)
WORKLOAD_AND_DATA_PATH=${WORKLOADS_AND_DATA_PATHS[$SLURM_ARRAY_TASK_ID]}
WORKLOAD="${WORKLOAD_AND_DATA_PATH% *}"
DATAPATH="${WORKLOAD_AND_DATA_PATH#* }"

echo "Running workload $WORKLOAD with data path $DATAPATH"

torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
         --standalone \
         --nnodes=1 \
         --nproc_per_node=4 \
         submission_runner.py \
         --framework=pytorch \
         --data_dir=data/$DATADIR/ \
         --experiment_dir=experiments/profile/ \
         --experiment_name=$WORKLOAD \
         --workload=$WORKLOAD \
         --submission_path=submissions/submission_folder/external_tuning/sirfshampoo/submission.py \
         --tuning_search_space=submissions/submission_folder/external_tuning/sirfshampoo/tuning_search_space.json \
         --overwrite=True  \
         --save_checkpoints=False \
         --profile \
         --max_global_steps=100
