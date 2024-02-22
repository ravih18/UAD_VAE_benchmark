from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
import nibabel as nib
from nilearn.image import resample_to_img

def load_session_list(maps_path, group):
    """
    Load all sub/session from tsv
    """
    if group == "train" or group == "validation":
        tsv_file = path.join(maps_path, "groups", group, f"split-{split}", "data.tsv")
    else:
        tsv_file = path.join(maps_path, "groups", group, "data.tsv")
    df = pd.read_csv(tsv_file, sep="\t", usecols=["participant_id", "session_id"])
    return list(df.to_records(index=False))


def mean_in_mask(X, mask):
    return X.mean(where=mask.astype(bool))


def healthiness_score(X, mask, out_mask):
    return mean_in_mask(X, mask) / mean_in_mask(X, out_mask)


def load_masks(pathology):
    """
    Load region mask
    """
    caps_dir = "/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform"
    img_path = path.join(caps_dir, 
                     "subjects", 
                     "sub-ADNI002S0685", 
                     "ses-M48", 
                     "t1_linear", 
                     "sub-ADNI002S0685_ses-M48_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz")
    mni_path = "/gpfswork/rech/krk/usy14zi/vae_benchmark/data/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"

    mask_path = path.join(caps_dir, "masks", f"mask_hypo_{pathology}.nii")

    mask_nii = nib.load(mask_path)
    mask_nii = resample_to_img(mask_nii, nib.load(img_path), interpolation='nearest')
    pathology_mask = mask_nii.get_fdata()

    # Load MNI template mask
    mni_nii = nib.load(mni_path)
    mni_nii = resample_to_img(mni_nii, nib.load(img_path), interpolation='nearest')
    brain_mask = mni_nii.get_fdata()
    out_mask = brain_mask - pathology_mask
    
    return pathology_mask, out_mask


def compute_metrics(maps_path, split, session, test_set, pathology_mask, reference_mask, columns):
    """
    """
    sub, ses = session[0], session[1]
    
    # Load all IO image
    gt_file = sub + "_" + ses + "_image-0_input.pt"
    gt_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
    gt_array = torch.load(gt_path).numpy()
    
    input_file = sub + "_" + ses + "_image-0_input.pt"
    input_path = path.join(maps_path, f"split-{split}", "best-loss", test_set, "tensors", input_file)
    input_array = torch.load(input_path).numpy()
    
    recon_file = sub + "_" + ses + "_image-0_output.pt"
    recon_path = path.join(maps_path, f"split-{split}", "best-loss", test_set, "tensors", recon_file)
    recon_array = torch.load(recon_path).detach().numpy()
    
    healthiness_gt = healthiness_score(gt_array, pathology_mask, reference_mask)
    healthiness_input = healthiness_score(input_array, pathology_mask, reference_mask)
    healthiness_recon = healthiness_score(recon_array, pathology_mask, reference_mask)
    
    test_set_label = ' '.join(test_set.upper().split('_')[2:])
    row1 = [sub, ses, test_set_label, "ground truth $x$", healthiness_gt]
    row2 = [sub, ses, test_set_label, "simulation $x'$", healthiness_input]
    row3 = [sub, ses, test_set_label, "reconstruction $\widehat{x'}$", healthiness_recon]

    return pd.DataFrame([row1, row2, row3], columns=columns.keys())


def make_healthiness_dataframe(maps_path, split, test_sets, columns):
    """
    """
    results_df = pd.DataFrame(columns) 

    for test_set in test_sets:

        sessions_list = load_session_list(maps_path, test_set)
        pathology = test_set.split("_")[-2]
        pathology_mask, out_mask = load_masks(pathology)
        
        for session in sessions_list:
            row_df = compute_metrics(maps_path, split, session, test_set, pathology_mask, out_mask, columns)
            results_df = pd.concat([results_df, row_df])
    return results_df

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_path')
    parser.add_argument('-s', '--split', default=0)
    args = parser.parse_args()

    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "test_set": pd.Series(dtype='str'),
        "measure":pd.Series(dtype='str'),
        "healthiness": pd.Series(dtype='float'),
    }

    test_sets_percentages = [
        "test_hypo_ad_5",
        "test_hypo_ad_10",
        "test_hypo_ad_15",
        "test_hypo_ad_20",
        "test_hypo_ad_30",
        "test_hypo_ad_40",
        "test_hypo_ad_50",
        "test_hypo_ad_70",
    ]
    df_percentage = make_healthiness_dataframe(args.maps_path, args.split, test_sets_percentages, columns)
    df_percentage.to_csv(path.join(args.maps_path, f"split-{args.split}", "best-loss/healthiness_percentage.tsv"), sep="\t", index=False)

    test_sets_pathologies = [
        "test_hypo_ad_30",
        "test_hypo_pca_30",
        "test_hypo_bvftd_30",
        "test_hypo_lvppa_30",
        "test_hypo_svppa_30",
        "test_hypo_nfvppa_30",
    ]
    df_pathology = make_healthiness_dataframe(args.maps_path, args.split, test_sets_pathologies, columns)
    df_pathology.to_csv(path.join(args.maps_path, f"split-{args.split}", "best-loss/healthiness_pathology.tsv"), sep="\t", index=False)
