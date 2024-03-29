#!/bin/bash3
#SBATCH --time=01:00:00
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --array=1-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

echo $1

ROOT_DIR=./random_search_benchmark
RS_DIR=${ROOT_DIR}/random_search_${1}

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'
clinicadl predict ${RS_DIR}/maps/MAPS_${1}_${SLURM_ARRAY_TASK_ID} validation
