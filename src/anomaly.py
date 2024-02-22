from os import path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import nibabel as nib


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


def create_mask(atlas, value):
    return (atlas==value).astype(int)


def anomaly_score(X, atlas, atlas_dict):
    score = {}
    # Iterate over the regions
    for region, value in atlas_dict.items():
        # make a mask of the region
        mask = create_mask(atlas, value)
        # compute the mean intensity within the region (region mean)
        rm = mean_in_mask(X, mask)
        score[region] = rm
    return score


def get_dataframe(maps_path, split, group):
    """
    """
    sessions_list = load_session_list(maps_path, group)

    # Load atlas
    atlas_gm = nib.load("/gpfswork/rech/krk/commun/anomdetect/atlas/AAL2/AAL2_mni_gm.nii").get_fdata()
    atlas_df = pd.read_csv("/gpfswork/rech/krk/commun/anomdetect/atlas/AAL2/AAL2_new_index.tsv", sep="\t")
    atlas_dict = dict(zip(atlas_df.Region, atlas_df.Value))
    regions = list(atlas_df.Region)
    regions.remove('Background')

    columns = {
        "participant_id": pd.Series(dtype='str'),
        "session_id": pd.Series(dtype='str'),
        "image": pd.Series(dtype='str'),
        "region": pd.Series(dtype='str'),
        "metric": pd.Series(dtype='float'),
    }
    df = pd.DataFrame(columns)
    
    def compute_metrics(session, atlas, atlas_dict):
        sub, ses = session[0], session[1]

        # Load all IO image
        input_file = sub + "_" + ses + "_image-0_input.pt"
        input_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", input_file)
        input_array = torch.load(input_path).numpy()

        gt_file = sub + "_" + ses + "_image-0_input.pt"
        gt_path = path.join(maps_path, f"split-{split}", "best-loss", "test_CN", "tensors", gt_file)
        gt_array = torch.load(gt_path).numpy()
        
        recon_file = sub + "_" + ses + "_image-0_output.pt"
        recon_path = path.join(maps_path, f"split-{split}", "best-loss", group, "tensors", recon_file)
        recon_array = torch.load(recon_path).detach().numpy()

        score_inp = anomaly_score(input_array, atlas, atlas_dict)
        score_gt = anomaly_score(gt_array, atlas, atlas_dict)
        score_rec = anomaly_score(recon_array, atlas, atlas_dict)
        return score_inp, score_gt, score_rec

    for session in sessions_list:
        score_inp, score_gt, score_rec = compute_metrics(session, atlas_gm, atlas_dict)
        for region in regions:
            row = pd.DataFrame([[session[0], session[1], "Simulated image", region, score_inp[region]]], columns=columns.keys())
            df = pd.concat([df, row])
            row = pd.DataFrame([[session[0], session[1], "Original image", region, score_gt[region]]], columns=columns.keys())
            df = pd.concat([df, row])
            row = pd.DataFrame([[session[0], session[1], "Network reconstruction", region, score_rec[region]]], columns=columns.keys())
            df = pd.concat([df, row])
    return df, regions


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('maps_path')
    parser.add_argument('-s', '--split', default=0)
    args = parser.parse_args()

    group = "test_hypo_ad_30"
    df, _ = get_dataframe(args.maps_path, args.split, group)
    df.to_csv(path.join(args.maps_path, f"split-{args.split}", "best-loss", group, "anomaly.tsv"), sep="\t", index=False)
