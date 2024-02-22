#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --array=1-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

echo $1

ROOT_DIR=./random_search_benchmark
RS_DIR=${ROOT_DIR}/random_search_${1}

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'
clinicadl random-search vae-architecture $RS_DIR random_search_${SLURM_ARRAY_TASK_ID}.toml ${RS_DIR}/maps/MAPS_${1}_${SLURM_ARRAY_TASK_ID}
