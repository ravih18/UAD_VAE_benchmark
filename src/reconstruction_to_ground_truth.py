from os import path
import pandas as pd
from torch import load
from tqdm import tqdm
import numpy as np


def mse_fn(y, y_pred):
    return np.mean(np.square(y - y_pred))

def psnr_fn(y, y_pred):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(y, y_pred)

def ssim_fn(y, y_pred):
    from clinicadl.utils.pytorch_ssim import ssim3D
    return ssim3D(y, y_pred).item()


def make_recon2gt_df(maps_directory, split):

    # Get test set session list
    tsv_file = path.join(maps_directory, "groups/test_CN/data.tsv")
    df_participants = pd.read_csv(tsv_file, sep="\t", usecols=["participant_id", "session_id"])
    sessions_list = list(df_participants.to_records(index=False))

    # loop on each group
    for group in groups:
        # Initialize output dataframe
        columns = {
            "participant_id": pd.Series(dtype='str'),
            "session_id": pd.Series(dtype='str'),
            "Test group": pd.Series(dtype='int'),
            "Metric": pd.Series(dtype='str'),
            "Value": pd.Series(dtype='float'),
        }
        results_df = pd.DataFrame(columns)
        for session in tqdm(sessions_list):
            sub, ses = session[0], session[1]
            #if sub != "sub-ADNI067S0257":
            # Load reconstruction
            recon_file = sub + "_" + ses + "_image-0_output.pt"
            recon_path = path.join(maps_directory, f"split-{split}", "best-loss", group, "tensors", recon_file)
            recon_array = load(recon_path).detach().unsqueeze(0).numpy()
            # Load ground truth image
            gt_file = sub + "_" + ses + "_image-0_input.pt"
            gt_path = path.join(maps_directory, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
            gt_array = load(gt_path).numpy()
            # for each session compute each metric
            results_df = pd.concat([results_df, pd.DataFrame([[sub, ses, group, "MSE", mse_fn(recon_array, gt_array)]], columns=columns.keys())])
            results_df = pd.concat([results_df, pd.DataFrame([[sub, ses, group, "SSIM", ssim_fn(recon_array, gt_array)]], columns=columns.keys())])
            results_df = pd.concat([results_df, pd.DataFrame([[sub, ses, group, "PSNR", psnr_fn(recon_array, gt_array)]], columns=columns.keys())])
        # write the results in a tsv file (for each groups)
        results_df.to_csv(path.join(maps_directory, f"split-{split}", "best-loss", group,"reconstruction_to_groundtruth.tsv"), sep='\t', index=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_path')
    parser.add_argument('-s', '--split', default=0)
    args = parser.parse_args()

    groups = [
        "test_hypo_ad_30",
        "test_hypo_bvftd_30",
        "test_hypo_pca_30",
        "test_hypo_lvppa_30",
        "test_hypo_nfvppa_30",
        "test_hypo_svppa_30"
    ]

    make_recon2gt_df(args.maps_path, args.split)
