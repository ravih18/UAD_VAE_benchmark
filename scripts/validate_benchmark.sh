#!/usr/bin/env bash

ROOT_DIR=./random_search_benchmark

MODELS=(Adversarial_AE BetaTCVAE BetaVAE DisentangledBetaVAE FactorVAE HVAE MSSSIM_VAE IWAE INFOVAE_MMD RAE_L2 RAE_GP SVAE VAEGAN VAE_IAF VAE_LinNF VAMP VQVAE WAE_MMD)
for MODEL in ${MODELS[@]} ; do
    echo Validation $MODEL
    LAUNCH_DIR=${ROOT_DIR}/random_search_${MODEL}
    sbatch --output=${LAUNCH_DIR}/logs/val_%j.log --job-name=val_${MODEL} ${LAUNCH_DIR}/validate_rs.sh $MODEL
done
