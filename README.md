# Pseudo-healthy image reconstruction with variational autoencoders for anomaly detection: A benchmark on 3D brain FDG PET

## Overview

This repository provides the source code and data necessary to run a benchmark of VAE based models trained to reconstruct pseudo-healthy images for unsupervised anomaly detection on 3D brain FDG PET.

The model has been trained using the [ClinicaDL](https://clinicadl.readthedocs.io/en/latest/) open source software, [Pythae](https://pythae.readthedocs.io/en/latest/models/pythae.models.html) python librairy, and using the [ADNI dataset](https://adni.loni.usc.edu/).

The method is described in the following article [Pseudo-healthy image reconstruction with variational autoencoders for anomaly detection: A benchmark on 3D brain FDG PET](https://hal.science/hal-04445378v1) (currently under revision).

If you use any ressources from this repository, please cite us:
```bibtex
@unpublished{hassanaly:hal-04445378,
  TITLE = {{Pseudo-healthy image reconstruction with variational autoencoders for anomaly detection: A benchmark on 3D brain FDG PET}},
  AUTHOR = {Hassanaly, Ravi and Solal, Ma{\"e}lys and Colliot, Olivier and Burgos, Ninon},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  KEYWORDS = {Variational autoencoder ; Unsupervised anomaly detection ; Deep generative models ; PET ; Alzheimer's disease},
}
```

## Requirement

### Environment

To come

### ADNI Dataset

In order to improve reproducibility, the ADNI dataset have been converted to [BIDS]() and preprocessed using Clinica open source software.

Once we obtain a BIDS of ADNI using the [`adni-to-bids`](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/ADNI2BIDS/) command, we run the [`pet-linear`](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/PET_Linear/) command on the dataset:
```
clinica run pet-linear $BIDS_DIRECTORY $CAPS_DIRECTORY 18FFDG cerebellumPons2
```
The outputs are stored in a CAPS dataset at $CAPS_DIRECTORY.

All the details on the data selection are in the Appendix of the article.

### Build test sets

To build the different simulated dataset used for our evaluation, run following ClinicaDL command with the different parameters of `$PATHOLOGY` and `$PERCENTAGE`:
```
clinicadl generate hypometabolic $CAPS_DIRECTORY $GENERATED_CAPS_DIRECTORY --pathology $PATHOLOGY --anomaly_degree $PERCENTAGE
```

All the different test sets build for the experiments are detailed in the article. The procedure was defined in a [previous work](https://www.melba-journal.org/papers/2024:003.html).

### VAE architecture random search

To come

### VAE variants benchmark

To run the benchaark of VAE variants, run the `scripts/benchmark.sh` SLURM scrit:
```
sbatch scripts/benchmark.sh
```

To run validation on all of the models:
```
sbatch scripts/validate_benchmark.sh
```

### Model training and split selection

Once the best hyperparameters are defined, train all the models on the K splits of the k-folds (paralleize on K nodes of the cluster):
```
sbatch split_selection/train_model.sh MODEL_NAME
```

To validate each model individually:
```
sbatch split_selection/validation.sh MODEL_NAME
```

Once the best split is selected, write them in the `split_selection/best_model_split.tkt`. It is then possible to lunch evaluation of the models using the test sets for all the models:
```
sh eval_benchmark.sh
```
Be careful, using the simulated test sets, this requires a lot of storage memory (around 300GB per model).