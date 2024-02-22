#!/usr/bin/env bash

while IFS= read -r line
do
    MODEL_SPLIT=($line)
    sbatch --job-name=eval_${MODEL_SPLIT[0]} final_models/evaluation.sh ${MODEL_SPLIT[0]} ${MODEL_SPLIT[1]}
done < ./final_models/best_model_split.txt
