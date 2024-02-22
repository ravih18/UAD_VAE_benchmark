#!/usr/bin/env bash

#MODEL_LIST=(BetaTCVAE VAE_IAF RAE_GP HVAE DisentangledBetaVAE FactorVAE WAE_MMD SVAE IWAE)
MODEL=IWAE

RS_DIR=./random_search_benchmark/random_search_${MODEL}

for file in ${RS_DIR}/logs/*; do
    echo $file
    if grep -q "DUE TO TIME LIMIT" "$file"; then
        sbatch --output=${RS_DIR}/logs/%j.log --job-name=resume_${MODEL} scripts/resume_training.sh $MODEL $file
    fi
done
