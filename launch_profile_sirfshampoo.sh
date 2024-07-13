#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=m4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-35

# NOTE This will use smaller batch sizes to fit on Vector's A40 GPUs.
# Set this value to 0 if you are using GPUs with more than 48 GiB RAM.
export RUNNING_ON_VECTOR_CLUSTER=1

# NOTE Make sure you have all data sets set up properly

WORKLOADS_AND_DATA_PATHS=(
    # "cifar TODO"
    # "criteo1tb TODO"
    # "criteo1tb_test TODO"
    # "criteo1tb_layernorm TODO"
    # "criteo1tb_embed_init TODO"
    # "criteo1tb_resnet TODO"
    "fastmri fastmri"
    "fastmri_model_size fastmri"
    "fastmri_tanh fastmri"
    "fastmri_layernorm fastmri"
    "imagenet_resnet imagenet"
    "imagenet_resnet_silu imagenet"
    "imagenet_resnet_gelu imagenet"
    "imagenet_resnet_large_bn_init imagenet"
    "imagenet_vit imagenet"
    "imagenet_vit_glu imagenet"
    "imagenet_vit_post_ln imagenet"
    "imagenet_vit_map imagenet"
    # "librispeech_conformer TODO"
    # "librispeech_conformer_attention_temperature TODO"
    # "librispeech_conformer_layernorm TODO"
    # "librispeech_conformer_gelu TODO"
    # "librispeech_deepspeech TODO"
    # "librispeech_deepspeech_tanh TODO"
    # "librispeech_deepspeech_no_resnet TODO"
    # "librispeech_deepspeech_norm_and_spec_aug TODO"
    # "mnist TODO"
    "ogbg ogbg"
    "ogbg_gelu ogbg"
    "ogbg_silu ogbg"
    "ogbg_model_size ogbg"
    "wmt wmt"
    "wmt_attention_temp wmt"
    "wmt_glu_tanh wmt"
)

if [ ${SLURM_ARRAY_TASK_ID} -ge ${#WORKLOADS_AND_DATA_PATHS[@]} ]; then
    echo "SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) is out of bounds."
    exit
fi

WORKLOAD_AND_DATA_PATH=${WORKLOADS_AND_DATA_PATHS[$SLURM_ARRAY_TASK_ID]}
WORKLOAD="${WORKLOAD_AND_DATA_PATH% *}"
DATAPATH="${WORKLOAD_AND_DATA_PATH#* }"

echo "Running workload $WORKLOAD assuming data is in data/$DATAPATH"

torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
         --standalone \
         --nnodes=1 \
         --nproc_per_node=4 \
         submission_runner.py \
         --framework=pytorch \
         --data_dir=data/$DATAPATH/ \
         --experiment_dir=experiments/profile_sirfshampoo/ \
         --experiment_name=$WORKLOAD \
         --workload=$WORKLOAD \
         --submission_path=submissions/submission_folder/external_tuning/sirfshampoo/submission.py \
         --tuning_search_space=submissions/submission_folder/external_tuning/sirfshampoo/tuning_search_space.json \
         --overwrite=True  \
         --save_checkpoints=False \
         --profile \
         --max_global_steps=100
