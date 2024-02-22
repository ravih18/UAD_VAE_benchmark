#!/bin/bash
#SBATCH --job-name=validation_model
#SBATCH --output=logs/val_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:30:00

echo $1

ROOT_DIR=/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/final_models
MAPS_DIR=${ROOT_DIR}/maps/MAPS_${1}
mv ${ROOT_DIR}/${1}/MAPS_${1}_0 $MAPS_DIR

clinicadl predict $MAPS_DIR validation
