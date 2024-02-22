#!/bin/bash
#SBATCH --output=final_models/logs/eval_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100

echo Model $1 split $2

MAPS_DIR=/gpfswork/rech/krk/commun/anomdetect/journal_benchmark/final_models/maps/MAPS_${1}
echo $MAPS_DIR

# # Predict on test AD
# GROUPE=test_AD
# CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform/
# PARTICIPANT_TSV=/gpfswork/rech/krk/commun/anomdetect/tsv_files/test/AD_baseline.tsv
# echo Predict test AD
# clinicadl  predict $MAPS_DIR $GROUPE \
#            --caps_directory $CAPS_DIR \
#            --participants_tsv $PARTICIPANT_TSV \
#            --diagnoses AD \
#            --split $2 \
#            --selection_metrics loss \
#            --overwrite

# Save tensors of reduced test AD
GROUPE=test_AD_reduced
CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform/
PARTICIPANT_TSV=/gpfswork/rech/krk/commun/anomdetect/tsv_files/test/AD_reduced_baseline.tsv
echo Predict reduced test AD
clinicadl  predict $MAPS_DIR $GROUPE \
           --caps_directory $CAPS_DIR \
           --participants_tsv $PARTICIPANT_TSV \
           --diagnoses AD \
           --split $2 \
           --selection_metrics loss \
           --save_tensor \
           --overwrite

# # Predict on test CN
# GROUPE=test_CN
# PARTICIPANT_TSV=/gpfswork/rech/krk/commun/anomdetect/tsv_files/CN-test_baseline.tsv
# echo Predict test CN
# clinicadl  predict $MAPS_DIR $GROUPE \
#            --caps_directory $CAPS_DIR \
#            --participants_tsv $PARTICIPANT_TSV \
#            --diagnoses CN \
#            --split $2 \
#            --selection_metrics loss \
#            --save_tensor \
#            --overwrite

# # Predict on all hypometabolic CAPS
# PATHOLOGY_LIST=(bvftd lvppa svppa nfvppa pca)
# for PATHOLOGY in ${PATHOLOGY_LIST[@]} ; do
#     GROUPE=test_hypo_${PATHOLOGY}_30
#     CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/hypometabolic_caps/caps_${PATHOLOGY}_30

#     echo Predict hypo $GROUPE
#     clinicadl   predict $MAPS_DIR $GROUPE \
#                 --caps_directory $CAPS_DIR \
#                 --participants_tsv $PARTICIPANT_TSV \
#                 --diagnoses CN \
#                 --split $2 \
#                 --selection_metrics loss \
#                 --save_tensor \
#                 --overwrite
# done

# PERCENTAGE_LIST=(5 10 15 20 25 30 40 50 70)
# for PERCENTAGE in ${PERCENTAGE_LIST[@]} ; do
#    GROUPE=test_hypo_ad_${PERCENTAGE}
#    CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/hypometabolic_caps/caps_ad_${PERCENTAGE}

#    echo Predict hypo $GROUPE
#    clinicadl   predict $MAPS_DIR $GROUPE \
#                --caps_directory $CAPS_DIR \
#                --participants_tsv $PARTICIPANT_TSV \
#                --diagnoses CN \
#                --split $2 \
#                --selection_metrics loss \
#                --save_tensor \
#                --overwrite
# done

#echo making reconstruction to ground truth tsv
#python python_scripts/reconstruction_to_ground_truth.py $MAPS_DIR -s $2

#echo making healthiness tsv
#python python_scripts/healthiness.py $MAPS_DIR -s $2

#echo making anomaly score tsv
#python python_scripts/anomaly.py $MAPS_DIR -s $2
