ROOT_DIR=./random_search_benchmark

MODELS=(Adversarial_AE BetaTCVAE BetaVAE DisentangledBetaVAE FactorVAE HVAE MSSSIM_VAE IWAE INFOVAE_MMD RAE_L2 RAE_GP SVAE VAEGAN VAE_IAF VAE_LinNF VAMP VQVAE WAE_MMD)
#MODELS=(PIWAE MIWAE CIWAE RHVAE PoincareVAE)
for MODEL in ${MODELS[@]} ; do
    echo Random Search $MODEL
    LAUNCH_DIR=${ROOT_DIR}/random_search_${MODEL}
    sbatch --output=${LAUNCH_DIR}/logs/%j.log --job-name=RS_${MODEL} ${LAUNCH_DIR}/launch_rs.sh $MODEL
done