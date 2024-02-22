#!/bin/bash
#SBATCH --output=logs/train_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=60:00:00

echo $1

ROOT_DIR=/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/final_models
CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform
MAPS_DIR=${ROOT_DIR}/${1}/MAPS_${1}_${SLURM_ARRAY_TASK_ID}
PREPROCESSING_JSON=${CAPS_DIR}/tensor_extraction/extract_pet_uniform_image.json
TSV_DIR=/gpfswork/rech/krk/commun/anomdetect/tsv_files/6_fold
CONFIG_TOML=${ROOT_DIR}/config_files/config_${1}.toml

clinicadl train pythae $CAPS_DIR $PREPROCESSING_JSON $TSV_DIR $MAPS_DIR -c $CONFIG_TOML -s $SLURM_ARRAY_TASK_ID

if [ $SLURM_ARRAY_TASK_ID = 0 ]
then
    python update_json.py $MAPS_DIR
fi

if [ $SLURM_ARRAY_TASK_ID != 0 ]
then
    cp -r ${MAPS_DIR}/split-${SLURM_ARRAY_TASK_ID} ${ROOT_DIR}/${1}/MAPS_${1}_0/
fi
